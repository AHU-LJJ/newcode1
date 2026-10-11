from dataclasses import dataclass
from pathlib import Path
import json
import random
import re

import pyarrow.parquet as pq


@dataclass(frozen=True)
class Document:
    title: str
    sentences: tuple[str, ...]
    source_ids: tuple[str, ...] = ()
    speakers: tuple[str, ...] = ()
    timestamp: str = ""


@dataclass(frozen=True)
class Question:
    """The only dataset object the agent may receive: no evaluation labels."""
    id: str
    question: str
    documents: tuple[Document, ...]
    dataset: str = "hotpotqa"
    conversation_id: str = ""


@dataclass(frozen=True)
class Example:
    public: Question
    answer: str
    supporting_facts: tuple[tuple[str, int], ...]
    kind: str
    qa_index: int | None = None
    unresolved_evidence: tuple[str, ...] = ()


def load_examples(path: Path, limit: int, seed: int, offset: int = 0) -> list[Example]:
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    table = pq.read_table(path)
    indices = list(range(len(table)))
    random.Random(seed).shuffle(indices)
    selected = indices[offset:offset + limit]
    if len(selected) != limit:
        raise ValueError("Requested sample exceeds dataset size")
    examples = []
    for row in table.take(selected).to_pylist():
        context = row["context"]
        docs = tuple(Document(t, tuple(s)) for t, s in zip(
            context["title"], context["sentences"], strict=True))
        if not docs or not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("Expected a labeled HotpotQA split with context")
        sp = row["supporting_facts"]
        examples.append(Example(
            Question(row["id"], row["question"], docs), row["answer"],
            tuple(zip(sp["title"], sp["sent_id"], strict=True)), row["type"]))
    return examples


def locomo_documents(conversation: dict) -> tuple[Document, ...]:
    """Only raw dialogue, timestamps and supplied image captions; no annotations."""
    docs = []
    sessions = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                      key=lambda key: int(key.split("_")[1]))
    for session in sessions:
        date = str(conversation.get(session + "_date_time", "unknown date"))
        turns = conversation[session]
        texts = []
        for turn in turns:
            text = f"[{date}] {turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += " [Image caption: " + str(turn["blip_caption"]) + "]"
            texts.append(text)
        docs.append(Document(f"{session} | {date}", tuple(texts),
                             tuple(t["dia_id"] for t in turns),
                             tuple(t["speaker"] for t in turns), date))
    if not docs:
        raise ValueError("LoCoMo conversation has no sessions")
    ids = [s for d in docs for s in d.source_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dialogue IDs in one LoCoMo conversation")
    return tuple(docs)


def resolve_evidence(raw: list[str], docs: tuple[Document, ...]):
    mapping = {sid: (doc.title, i) for doc in docs for i, sid in enumerate(doc.source_ids)}
    pairs, unresolved = set(), []
    for value in raw:
        # Handle published formatting glitches: combined IDs, D:11:26, D30:05.
        matches = re.findall(r"D:?(\d+):(\d+)", value)
        if not matches:
            unresolved.append(value)
        for session, turn in matches:
            sid = f"D{int(session)}:{int(turn)}"
            if sid in mapping:
                pairs.add(mapping[sid])
            else:
                unresolved.append(sid)
    return tuple(sorted(pairs)), tuple(unresolved)


def load_locomo(path: Path, limit: int, seed: int, offset: int = 0,
                categories: tuple[int, ...] = (1, 2, 3, 4)) -> list[Example]:
    if limit < 1 or offset < 0 or not categories or not set(categories) <= {1, 2, 3, 4, 5}:
        raise ValueError("Invalid LoCoMo sample size, offset or categories")
    groups = {category: [] for category in sorted(set(categories))}
    for sample in json.loads(path.read_text()):
        docs = locomo_documents(sample["conversation"])
        for qa_index, qa in enumerate(sample["qa"]):
            category = qa["category"]
            if category not in groups:
                continue
            pairs, unresolved = resolve_evidence(qa.get("evidence", []), docs)
            answer = "no information available" if category == 5 else str(qa["answer"])
            groups[category].append(Example(
                Question(f"{sample['sample_id']}:qa:{qa_index}", qa["question"], docs,
                         dataset="locomo", conversation_id=sample["sample_id"]),
                answer, pairs, str(category), qa_index, unresolved))
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    # Category-balanced interleaving, independent of labels' content and evidence.
    ordered = []
    for i in range(max(map(len, groups.values()), default=0)):
        for group in groups.values():
            if i < len(group):
                ordered.append(group[i])
    selected = ordered[offset:offset + limit]
    if len(selected) != limit:
        raise ValueError("Requested sample exceeds eligible LoCoMo questions")
    return selected
