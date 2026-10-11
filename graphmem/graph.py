"""An independently implemented provenance graph; no PGMem code dependency."""
from collections import Counter
from hashlib import sha256
import math
import re

from .data import Question


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def rank(query: str, items: list[dict], limit: int) -> list[dict]:
    """Small-corpus BM25, stable ties. No embeddings or external service."""
    tokens = [Counter(words(x["text"])) for x in items]
    lengths = [sum(t.values()) for t in tokens]
    average = sum(lengths) / max(1, len(lengths)) or 1
    terms = set(words(query))
    frequency = {w: sum(w in t for t in tokens) for w in terms}
    scores = []
    for item, counts, length in zip(items, tokens, lengths):
        score = 0.0
        for term in terms:
            tf = counts[term]
            idf = math.log(1 + (len(items) - frequency[term] + .5) / (frequency[term] + .5))
            score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / average))
        if score > 0:
            scores.append((score, item))
    scores.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [dict(item, score=round(score, 5)) for score, item in scores[:limit]]


class MemoryGraph:
    def __init__(self, question: Question, ranker=None):
        self.ranker = ranker or rank
        self.nodes: dict[str, dict] = {}
        self.edges: set[tuple[str, str, str]] = set()
        self.events: list[dict] = []
        self.seen: set[str] = set()
        self.next_fact = 0
        for i, doc in enumerate(question.documents):
            did = f"d{i}"
            self.nodes[did] = dict(id=did, kind="document", text=doc.title, title=doc.title)
            for j, sentence in enumerate(doc.sentences):
                sid = f"s{i}:{j}"
                self.nodes[sid] = dict(id=sid, kind="sentence", text=sentence,
                                       title=doc.title, sent_id=j, document=did)
                if doc.source_ids:
                    self.nodes[sid]["source_id"] = doc.source_ids[j]
                if doc.timestamp:
                    self.nodes[sid]["timestamp"] = doc.timestamp
                if doc.speakers:
                    self.nodes[sid]["speaker"] = doc.speakers[j]
                    self.edges.add((self._entity(doc.speakers[j]), "speaker_of", sid))
                self.edges.add((did, "contains", sid))
                if j:
                    self.edges.add((f"s{i}:{j-1}", "next", sid))
            if question.dataset == "locomo" and i:
                self.edges.add((f"d{i-1}", "next_session", did))
        # Only exact title mentions, never gold supporting-fact edges.
        documents = [n for n in self.nodes.values() if n["kind"] == "document"]
        sentences = [n for n in self.nodes.values() if n["kind"] == "sentence"]
        for doc in documents:
            pattern = re.compile(r"(?<!\w)" + re.escape(doc["title"]) + r"(?!\w)", re.I)
            for sentence in sentences:
                if sentence["document"] != doc["id"] and pattern.search(sentence["text"]):
                    self.edges.add((sentence["id"], "mentions", doc["id"]))

    def documents(self) -> list[dict]:
        return [dict(n) for n in self.nodes.values() if n["kind"] == "document"]

    def reveal(self, nodes: list[dict]) -> list[dict]:
        self.seen.update(n["id"] for n in nodes if n["kind"] == "sentence")
        return nodes

    def read_document(self, document_id: str, start: int = 0, count: int = 8) -> dict:
        if document_id not in self.nodes or self.nodes[document_id]["kind"] != "document":
            raise ValueError("Unknown document ID")
        if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 16:
            raise ValueError("start >= 0 and 1 <= count <= 16 required")
        nodes = [dict(n) for n in self.nodes.values() if n.get("document") == document_id]
        return dict(sentences=self.reveal(nodes[start:start + count]), total=len(nodes),
                    next_start=start + count if start + count < len(nodes) else None)

    def search(self, query: str, limit: int = 6, graph: bool = False) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("query must be 1..1000 characters")
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("limit must be 1..16")
        kinds = {"sentence", "fact", "entity", "document"} if graph else {"sentence"}
        candidates = [dict(n, text=(n.get("title", "") + " " + n["text"]).strip())
                      for n in self.nodes.values() if n["kind"] in kinds and n.get("active", True)]
        hits = self.ranker(query, candidates, limit)
        return {"nodes": self.reveal(hits)}

    def expand(self, node_id: str, hops: int = 1, limit: int = 16) -> dict:
        if node_id not in self.nodes or not self.nodes[node_id].get("active", True):
            raise ValueError("Unknown or retracted node")
        if type(hops) is not int or hops not in (1, 2) or type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("hops must be 1 or 2; limit must be 1..32")
        selected = [node_id]
        visited = {node_id}
        frontier = [node_id]
        truncated = False
        for _ in range(hops):
            following = []
            for current in frontier:
                for source, relation, target in sorted(self.edges):
                    if current not in (source, target):
                        continue
                    neighbor = target if current == source else source
                    if neighbor in visited or not self.nodes[neighbor].get("active", True):
                        continue
                    if len(selected) >= limit:
                        truncated = True
                        continue
                    visited.add(neighbor)
                    selected.append(neighbor)
                    following.append(neighbor)
            frontier = following
        edges = [list(e) for e in sorted(self.edges) if e[0] in visited and e[2] in visited]
        return dict(nodes=self.reveal([dict(self.nodes[n]) for n in selected]),
                    edges=edges, truncated=truncated)

    def _validate_fact(self, subject: str, relation: str, object: str, evidence: list[str]) -> None:
        for value in (subject, relation, object):
            if not isinstance(value, str) or not value.strip() or len(value) > 500:
                raise ValueError("Fact fields must be nonempty strings up to 500 characters")
        if not isinstance(evidence, list) or not evidence or len(evidence) > 16:
            raise ValueError("1..16 evidence sentence IDs required")
        if any(not isinstance(e, str) or e not in self.seen for e in evidence):
            raise ValueError(self.evidence_help())

    def evidence_help(self) -> str:
        return ("Invalid evidence ID. Copy the sentence node's 'id' exactly, e.g. s0:0; "
                "do not append ':sent_id', use a document ID, or invent IDs. "
                "Already read sentence IDs: " + ", ".join(sorted(self.seen)))

    def _entity(self, label: str) -> str:
        key = " ".join(label.casefold().split())
        eid = "e" + sha256(key.encode()).hexdigest()[:16]
        self.nodes.setdefault(eid, dict(id=eid, kind="entity", text=label))
        return eid

    def _attach(self, fid: str) -> None:
        fact = self.nodes[fid]
        self.edges = {e for e in self.edges if fid not in (e[0], e[2])}
        self.edges.add((self._entity(fact["subject"]), "subject_of", fid))
        self.edges.add((fid, "object", self._entity(fact["object"])))
        for sid in fact["evidence"]:
            self.edges.add((fid, "supported_by", sid))

    def add_fact(self, subject: str, relation: str, object: str, evidence: list[str]) -> dict:
        self._validate_fact(subject, relation, object, evidence)
        if sum(n["kind"] == "fact" and n["active"] for n in self.nodes.values()) >= 64:
            raise ValueError("Active fact capacity reached (64); revise or retract first")
        fid = f"f{self.next_fact}"
        self.next_fact += 1
        self.nodes[fid] = dict(id=fid, kind="fact", subject=subject, relation=relation,
                              object=object, evidence=sorted(set(evidence)), active=True,
                              text=f"{subject} --{relation}--> {object}", version=1,
                              epistemic_status="model_assertion_with_source_citations")
        self._attach(fid)
        self.events.append(dict(action="add", after=dict(self.nodes[fid])))
        return dict(self.nodes[fid])

    def _fact(self, fact_id: str) -> dict:
        node = self.nodes.get(fact_id, {})
        if node.get("kind") != "fact" or not node.get("active"):
            raise ValueError("Unknown or retracted fact ID")
        return node

    def revise_fact(self, fact_id: str, subject: str, relation: str, object: str,
                    evidence: list[str], reason: str) -> dict:
        self._validate_fact(subject, relation, object, evidence)
        self._reason(reason)
        fact = self._fact(fact_id)
        before = dict(fact)
        fact.update(subject=subject, relation=relation, object=object,
                    evidence=sorted(set(evidence)), text=f"{subject} --{relation}--> {object}",
                    version=fact["version"] + 1)
        self._attach(fact_id)
        self.events.append(dict(action="revise", before=before, after=dict(fact), reason=reason))
        return dict(fact)

    @staticmethod
    def _reason(reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("A reason of 1..1000 characters is required")

    def retract_fact(self, fact_id: str, reason: str) -> dict:
        self._reason(reason)
        fact = self._fact(fact_id)
        fact["active"] = False
        self.events.append(dict(action="retract", fact_id=fact_id, reason=reason))
        return dict(fact)

    def supporting_pairs(self, evidence: list[str]) -> list[list]:
        if not isinstance(evidence, list) or any(not isinstance(e, str) or e not in self.seen for e in evidence):
            raise ValueError(self.evidence_help())
        return [[self.nodes[s]["title"], self.nodes[s]["sent_id"]] for s in sorted(set(evidence))]

    def snapshot(self) -> dict:
        return dict(nodes=list(self.nodes.values()), edges=[list(e) for e in sorted(self.edges)],
                    events=self.events, seen=sorted(self.seen))
