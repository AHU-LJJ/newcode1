"""Resume an unchanged experiment after 429s, retaining failed attempts and pacing transport.

This changes request timing only. It does not alter the graph policy, prompts or sample.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from graphmem.api import APIError, ChatClient
from graphmem.__main__ import main, save_json


def resume():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=7)
    args = parser.parse_args()
    if not 0 < args.interval <= 60:
        parser.error("interval must be in (0, 60] seconds")
    out = args.output.resolve()
    config = json.loads((out / "manifest.json").read_text())["config"]
    current_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
        for p in sorted((ROOT / "graphmem").glob("*.py")))).hexdigest()
    if config["code_sha256"] != current_hash:
        raise ValueError("Policy code changed; retry this run using its source snapshot instead")
    attempt_dir = out / "attempts"
    attempt_dir.mkdir(exist_ok=True)
    archived = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for path in sorted((out / "episodes").glob("*.json")):
        row = json.loads(path.read_text())
        if row["status"] == "api_error" and "429" in (row.get("error") or ""):
            target = attempt_dir / f"{path.stem}.{stamp}.json"
            path.replace(target)
            archived.append(target.name)
    log = dict(started_at=datetime.now(timezone.utc).isoformat(),
               code_sha256=current_hash, minimum_call_interval_seconds=args.interval,
               archived_failed_attempts=archived, retry_events=[])
    log_path = out / f"transport_resume_{stamp}.json"
    save_json(log_path, log)
    original = ChatClient.complete
    last_call = [0.0]

    def paced(self, messages, tools):
        for attempt in range(3):
            delay = args.interval - (time.monotonic() - last_call[0])
            if delay > 0:
                time.sleep(delay)
            last_call[0] = time.monotonic()
            try:
                return original(self, messages, tools)
            except APIError as exc:
                if "429" not in str(exc) or attempt == 2:
                    raise
                log["retry_events"].append(dict(status=429, cooldown_seconds=40,
                    timestamp=datetime.now(timezone.utc).isoformat()))
                save_json(log_path, log)
                print("HTTP 429: cooling down for 40 seconds before retry", flush=True)
                time.sleep(40)

    ChatClient.complete = paced
    argv = ["graphmem", "run", "--dataset", config["dataset"], "--output", str(out), "--resume"]
    for key in ("limit", "seed", "offset", "max_rounds", "max_tools", "max_tokens", "max_input_chars", "temperature", "timeout"):
        argv += ["--" + key.replace("_", "-"), str(config[key])]
    argv += ["--modes", *config["modes"], "--retrieval", config["retrieval"]]
    if config.get("categories"):
        argv += ["--categories", *map(str, config["categories"])]
    embedding = config.get("embedding")
    if embedding:
        argv += ["--embedding-backend", embedding["backend"], "--embedding-model", embedding["model"],
                 "--embedding-batch-size", str(embedding["batch_size"])]
        if embedding["backend"] == "local":
            argv += ["--embedding-device", embedding["device"], "--embedding-max-length", str(embedding["max_length"])]
    sys.argv = argv
    try:
        return main()
    finally:
        ChatClient.complete = original
        log["ended_at"] = datetime.now(timezone.utc).isoformat()
        save_json(log_path, log)


if __name__ == "__main__":
    raise SystemExit(resume())
