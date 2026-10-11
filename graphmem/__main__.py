import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

from .agent import MODES, run_agent
from .api import ChatClient, Settings
from .data import load_examples, load_locomo
from .embeddings import APIBGE, LocalBGE, CachedEncoder, DenseRanker, EmbeddingError
from .graph import MemoryGraph
from .metrics import score, score_locomo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets/hotpot_qa/distractor/validation-00000-of-00001.parquet"
LOCOMO_DATA = ROOT / "datasets/locomo/data/locomo10.json"


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summarize(records, modes, planned):
    summary = {"planned_per_mode": planned, "modes": {}, "paired_deltas": {}}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        if not rows:
            continue
        keys = sorted({key for r in rows for key in r["metrics"]})
        metric_counts = {key: sum(key in r["metrics"] for r in rows) for key in keys}
        metrics = {key: sum(r["metrics"].get(key, 0) for r in rows) / metric_counts[key] for key in keys}
        tools = Counter()
        for row in rows:
            tools.update(row["tool_counts"])
        summary["modes"][mode] = dict(
            completed=len(rows), successful=sum(r["status"] == "ok" for r in rows),
            status_counts=dict(Counter(r["status"] for r in rows)), metrics=metrics, metric_counts=metric_counts,
            category_scores={kind: dict(n=sum(r.get("type") == kind for r in rows),
                f1=sum(r["metrics"]["f1"] for r in rows if r.get("type") == kind) /
                   sum(r.get("type") == kind for r in rows))
                for kind in sorted({r["type"] for r in rows if "type" in r})},
            api_calls=sum(r["api_calls"] for r in rows),
            tokens={k: sum(r["tokens"].get(k, 0) for r in rows)
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            usage_complete=all(r["usage_complete"] for r in rows), tool_counts=dict(tools),
            tool_errors=sum(r["tool_errors"] for r in rows),
            memory_diagnostics={k: sum(r.get("memory_diagnostics", {}).get(k, 0) for r in rows)
                                for k in ("mutations", "active_facts", "derived_fact_reads", "source_sentences_seen")},
            mean_latency_seconds=sum(r["latency_seconds"] for r in rows) / len(rows))
    managed = {r["id"]: r for r in records if r["mode"] == "managed_graph"}
    for baseline in ("full_context", "flat", "static_graph"):
        base = {r["id"]: r for r in records if r["mode"] == baseline}
        common = sorted(managed.keys() & base.keys())
        if not common:
            continue
        comparison = {"n": len(common)}
        for metric in ("em", "f1", "sp_f1", "joint_f1"):
            eligible = [i for i in common if metric in managed[i]["metrics"] and metric in base[i]["metrics"]]
            if not eligible:
                continue
            clusters = {}
            for i in eligible:
                cluster = managed[i].get("conversation_id") or i
                clusters.setdefault(cluster, []).append(managed[i]["metrics"][metric] - base[i]["metrics"][metric])
            differences = [d for group in clusters.values() for d in group]
            rng = random.Random(2026)
            draws = []
            for _ in range(2000):
                draw = [d for group in rng.choices(list(clusters.values()), k=len(clusters)) for d in group]
                draws.append(sum(draw) / len(draw))
            draws.sort()
            comparison[metric] = dict(mean=sum(differences) / len(differences), n=len(differences),
                                      resampling_units=len(clusters),
                                      bootstrap_unit="conversation" if managed[eligible[0]].get("conversation_id") else "question",
                                      bootstrap_95_ci=[draws[49], draws[1949]])
        summary["paired_deltas"][f"managed_graph-minus-{baseline}"] = comparison
    return summary


def write_report(out, records, modes, count, dataset="hotpotqa"):
    summary = summarize(records, modes, count)
    save_json(out / "summary.json", summary)
    lines = [f"# {dataset} pilot", "", "Scores include failed episodes as empty predictions. "
             "Small samples are exploratory; intervals do not cover model nondeterminism.", "",
             "| Mode | Completed/planned | Successful | Answer EM | Answer F1 | SP F1 | Joint F1 | Reported tokens |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode, row in summary["modes"].items():
        m = row["metrics"]
        evidence_f1 = f"{m['sp_f1']:.3f}" if "sp_f1" in m else "N/A"
        joint_f1 = f"{m['joint_f1']:.3f}" if "joint_f1" in m else "N/A"
        lines.append(f"| {mode} | {row['completed']}/{count} | {row['successful']} | "
                     f"{m['em']:.3f} | {m['f1']:.3f} | {evidence_f1} | {joint_f1} | "
                     f"{row['tokens']['total_tokens']} |")
        selected = [r for r in records if r["mode"] == mode]
        save_json(out / f"predictions_{mode}.json", dict(
            answer={r["id"]: r["prediction"]["answer"] for r in selected},
            sp={r["id"]: r["prediction"]["supporting_facts"] for r in selected}))
    if dataset == "locomo":
        lines += ["", "LoCoMo F1 follows the category-aware official evaluator. EM and evidence F1 are auxiliary; "
                  "Joint F1 is not reported. Evidence scores exclude examples with unresolved/absent annotations. "
                  "See metric_counts. Bootstrap samples conversations, not independent questions.", "",
                  "| Mode | Category | N | Answer score |", "| --- | --- | --- | --- |"]
        for mode, row in summary["modes"].items():
            for category, values in row["category_scores"].items():
                lines.append(f"| {mode} | {category} | {values['n']} | {values['f1']:.3f} |")
    lines += ["", "Tool counts, failures, costs and paired bootstrap deltas: [summary.json](summary.json).",
              "", "Per-episode evidence graphs, tool traces and API-visible messages: `episodes/`.",
              "Gold labels are stored separately in `gold.json` and never passed to the agent."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def make_encoder(args, settings):
    if args.retrieval == "bm25":
        return None
    if args.embedding_backend == "local":
        backend = LocalBGE(args.embedding_local_path, args.embedding_device,
                           args.embedding_batch_size, args.embedding_max_length)
    else:
        # Reuse validated Chat API settings unless separate embedding credentials are supplied.
        key = os.getenv("EMBEDDING_API_KEY") or settings.key
        base = os.getenv("EMBEDDING_BASE_URL") or settings.base_url
        from urllib.parse import urlparse
        url = urlparse(base)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment or url.username or url.password:
            raise ValueError("Invalid EMBEDDING_BASE_URL")
        backend = APIBGE(key, base, args.embedding_model, args.embedding_batch_size)
    return CachedEncoder(backend, args.embedding_cache)


def record_embedding_usage(out, encoder, resume):
    if encoder and (encoder.hits or encoder.misses):
        filename = "embedding_usage_resume.json" if resume else "embedding_usage.json"
        save_json(out / filename, encoder.snapshot())


def main():
    parser = argparse.ArgumentParser(description="Tool-managed graph memory on local HotpotQA or LoCoMo")
    parser.add_argument("command", choices=["inspect", "run"])
    parser.add_argument("--dataset", choices=["hotpotqa", "locomo"], default="hotpotqa")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--categories", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4])
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--retrieval", choices=["bge-m3", "bm25"], default="bge-m3")
    parser.add_argument("--embedding-backend", choices=["local", "api"], default="local")
    parser.add_argument("--embedding-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--embedding-model", choices=["BAAI/bge-m3", "Pro/BAAI/bge-m3"], default="BAAI/bge-m3")
    parser.add_argument("--embedding-local-path", type=Path, default=ROOT / "models/bge-m3")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / ".cache/embeddings.sqlite")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-tools", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-input-chars", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.data = args.data or (LOCOMO_DATA if args.dataset == "locomo" else DEFAULT_DATA)
    args.modes = args.modes or (list(MODES[1:]) if args.dataset == "locomo" else list(MODES))
    for key in ("max_rounds", "max_tools", "max_tokens", "max_input_chars", "timeout", "embedding_batch_size", "embedding_max_length"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate modes")
    if args.embedding_max_length > 8192:
        parser.error("BGE-M3 supports at most 8192 tokens")
    examples = (load_locomo(args.data, args.limit, args.seed, args.offset, tuple(args.categories))
                if args.dataset == "locomo" else load_examples(args.data, args.limit, args.seed, args.offset))
    if args.command == "inspect":
        print(json.dumps(dict(path=str(args.data), sampled=len(examples),
                              types=dict(Counter(e.kind for e in examples)),
                              first_public=(asdict(examples[0].public) if args.dataset == "hotpotqa" else
                                  dict(question=examples[0].public.question,
                                       conversation_id=examples[0].public.conversation_id,
                                       sessions=len(examples[0].public.documents),
                                       turns=sum(len(d.sentences) for d in examples[0].public.documents),
                                       first_two_turns=examples[0].public.documents[0].sentences[:2])),
                              unresolved_evidence={e.public.id: e.unresolved_evidence for e in examples if e.unresolved_evidence},
                              first_graph_nodes=len(MemoryGraph(examples[0].public).nodes)),
                         ensure_ascii=False, indent=2))
        return 0
    settings = Settings.load(args.env)
    encoder = make_encoder(args, settings)
    ranker = DenseRanker(encoder) if encoder else None
    code_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                               for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    config = dict(dataset=args.dataset, data_sha256=digest_file(args.data), model=settings.model,
                  categories=sorted(set(args.categories)) if args.dataset == "locomo" else None,
                  retrieval=args.retrieval, embedding=encoder.metadata if encoder else None,
                  endpoint_sha256=hashlib.sha256(settings.base_url.encode()).hexdigest(),
                  code_sha256=code_hash, modes=args.modes, seed=args.seed, offset=args.offset,
                  limit=args.limit, sample_ids=[e.public.id for e in examples],
                  max_rounds=args.max_rounds, max_tools=args.max_tools, max_tokens=args.max_tokens,
                  max_input_chars=args.max_input_chars, temperature=args.temperature, timeout=args.timeout)
    out = args.output or ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output is not empty; choose another directory or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text())["config"] != config:
            raise ValueError("Resume rejected: dataset, sample, code, model or configuration changed")
    else:
        save_json(manifest, dict(created_at=datetime.now(timezone.utc).isoformat(), config=config))
    (out / "episodes").mkdir(exist_ok=True)
    save_json(out / "gold.json", [dict(_id=e.public.id, answer=e.answer,
                                     supporting_facts=e.supporting_facts, type=e.kind,
                                     unresolved_evidence=e.unresolved_evidence,
                                     conversation_id=e.public.conversation_id, qa_index=e.qa_index) for e in examples])
    records = []
    completed = set()
    for path in sorted((out / "episodes").glob("*.json")):
        row = json.loads(path.read_text())
        records.append(row)
        completed.add((row["id"], row["mode"]))
    client = ChatClient(settings, args.max_tokens, args.timeout, args.temperature)
    print(f"Model: {settings.model}; {len(examples)} questions x {len(args.modes)} modes; output: {out}", flush=True)
    failed = False
    try:
        if encoder and any((e.public.id, mode) not in completed for e in examples for mode in args.modes if mode != "full_context"):
            texts = set()
            for example in examples:
                if not any((example.public.id, mode) not in completed for mode in args.modes if mode != "full_context"):
                    continue
                graph = MemoryGraph(example.public)
                texts.update((n.get("title", "") + " " + n["text"]).strip() for n in graph.nodes.values())
            print(f"Preparing BGE-M3 index: {len(texts)} unique source texts (shared vector cache, isolated graphs)", flush=True)
            encoder.encode(sorted(texts))
            record_embedding_usage(out, encoder, args.resume)
            print("Source index ready", flush=True)
        for index, example in enumerate(examples):
            # Deterministic rotation avoids always running one condition first.
            shift = index % len(args.modes)
            for mode in args.modes[shift:] + args.modes[:shift]:
                if (example.public.id, mode) in completed:
                    continue
                row = run_agent(example.public, mode, client, args.max_rounds, args.max_tools, args.max_input_chars, ranker)
                row["type"] = example.kind
                row["dataset"] = args.dataset
                row["conversation_id"] = example.public.conversation_id
                row["qa_index"] = example.qa_index
                if args.dataset == "locomo":
                    source_ids = {(doc.title, i): sid for doc in example.public.documents for i, sid in enumerate(doc.source_ids)}
                    row["prediction"]["dialogue_ids"] = [source_ids[tuple(pair)] for pair in row["prediction"]["supporting_facts"]]
                    row["metrics"] = score_locomo(row["prediction"], example)
                else:
                    row["metrics"] = score(row["prediction"]["answer"], row["prediction"]["supporting_facts"],
                                           example.answer, example.supporting_facts)
                filename = hashlib.sha256(example.public.id.encode()).hexdigest()[:24] + "_" + mode + ".json"
                save_json(out / "episodes" / filename, row)
                records.append(row)
                write_report(out, records, args.modes, len(examples), args.dataset)
                record_embedding_usage(out, encoder, args.resume)
                print(f"[{index + 1}/{len(examples)}] {mode}: {row['status']} "
                      f"EM={row['metrics']['em']:.0f} F1={row['metrics']['f1']:.3f} "
                      f"calls={row['api_calls']} tools={row['tool_counts']}", flush=True)
                if row["status"] in {"api_error", "protocol_error", "embedding_error"}:
                    print(row["error"], file=sys.stderr)
                    failed = True
                    break
            if failed:
                break
    finally:
        client.close()
        if encoder:
            # Preserve original usage on a no-op resume, rather than overwrite with zero new work.
            record_embedding_usage(out, encoder, args.resume)
            encoder.close()
        write_report(out, records, args.modes, len(examples), args.dataset)
    print(f"Report: {out / 'report.md'}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, EmbeddingError) as exc:
        print(f"Configuration/data error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted; completed episodes and report are saved. Use --resume with the same configuration.", file=sys.stderr)
        raise SystemExit(130)
