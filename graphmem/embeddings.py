"""BGE-M3 dense embeddings, cosine retrieval, and content-addressed vector caching."""
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

import httpx
import numpy as np

DIMENSION = 1024


class EmbeddingError(RuntimeError):
    pass


def normalize_vectors(values, count):
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.shape != (count, DIMENSION) or not np.isfinite(vectors).all():
        raise EmbeddingError("BGE-M3 must return finite 1024-dimensional vectors for every input")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise EmbeddingError("Embedding backend returned a zero vector")
    return vectors / norms


class LocalBGE:
    def __init__(self, path: Path, device="auto", batch_size=8, max_length=1024):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not any((path / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
            raise EmbeddingError(f"Local BGE-M3 weights missing at {path}; download first or use --embedding-backend api")
        revision_path = path / "source_revision.json"
        if not revision_path.exists():
            raise EmbeddingError("Local model needs source_revision.json to identify its official model revision")
        source = json.loads(revision_path.read_text())
        if source.get("repo_id") != "BAAI/bge-m3" or not source.get("revision"):
            raise EmbeddingError("Local model provenance must identify BAAI/bge-m3 and its revision")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingError("CUDA is unavailable in this process; run with GPU access or select cpu/api explicitly")
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False, weights_only=True,
            dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.metadata = dict(backend="local", model="BAAI/bge-m3", revision=source["revision"],
                             pooling="cls", normalized=True, dimension=DIMENSION,
                             dtype="float16" if device == "cuda" else "float32", device=device,
                             max_length=max_length, batch_size=batch_size)
        self.stats = dict(batches=0, texts=0, input_tokens=0, seconds=0.0)

    def encode(self, texts):
        import torch
        started = time.monotonic()
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.max_length:
                raise EmbeddingError("Embedding input exceeds max_length; increase --embedding-max-length or split sources (no silent truncation)")
            self.stats["input_tokens"] += int(tokens["attention_mask"].sum())
            try:
                with torch.inference_mode():
                    hidden = self.model(**{k: v.to(self.device) for k, v in tokens.items()}).last_hidden_state
                    outputs.append(hidden[:, 0].float().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise EmbeddingError("CUDA out of memory; reduce --embedding-batch-size or use the API backend") from None
            self.stats["batches"] += 1
            self.stats["texts"] += len(batch)
        self.stats["seconds"] += time.monotonic() - started
        return normalize_vectors(np.concatenate(outputs), len(texts))

    def close(self):
        pass


class APIBGE:
    def __init__(self, key, base_url, model="BAAI/bge-m3", batch_size=8):
        if model not in {"BAAI/bge-m3", "Pro/BAAI/bge-m3"}:
            raise EmbeddingError("This experiment requires BAAI/bge-m3 or Pro/BAAI/bge-m3")
        self.key, self.url, self.model = key, base_url.rstrip("/") + "/embeddings", model
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=90, follow_redirects=False)
        self.metadata = dict(backend="api", model=model, dimension=DIMENSION, normalized=True,
                             endpoint_sha256=sha256(base_url.encode()).hexdigest(), revision="provider_unspecified",
                             batch_size=batch_size)
        self.stats = dict(batches=0, texts=0, input_tokens=0, seconds=0.0, http_requests=0, usage_complete=True)

    def encode(self, texts):
        started = time.monotonic()
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            for attempt in range(3):
                self.stats["http_requests"] += 1
                try:
                    response = self.client.post(self.url,
                        headers={"Authorization": "Bearer " + self.key},
                        json=dict(model=self.model, input=batch, encoding_format="float"))
                except httpx.TransportError:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise EmbeddingError("Embedding API transport failure; credentials and URL omitted") from None
                if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code != 200:
                    raise EmbeddingError(f"Embedding API HTTP {response.status_code}; no automatic model substitution")
                break
            try:
                data = response.json()
                entries = sorted(data["data"], key=lambda e: e["index"])
                if [e["index"] for e in entries] != list(range(len(batch))):
                    raise ValueError
                outputs.append(normalize_vectors([e["embedding"] for e in entries], len(batch)))
                usage = data.get("usage", {}).get("total_tokens")
                if type(usage) is int:
                    self.stats["input_tokens"] += usage
                else:
                    self.stats["usage_complete"] = False
            except (ValueError, TypeError, KeyError):
                raise EmbeddingError("Malformed embedding API response") from None
            self.stats["batches"] += 1
            self.stats["texts"] += len(batch)
        self.stats["seconds"] += time.monotonic() - started
        return np.concatenate(outputs)

    def close(self):
        self.client.close()


class CachedEncoder:
    """Only vectors are shared across questions, never graph facts or answer state."""
    def __init__(self, backend, path: Path):
        self.backend = backend
        self.metadata = backend.metadata
        self.namespace = sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        self.hits = self.misses = 0

    def encode(self, texts):
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)
        keys = [sha256((self.namespace + "\0" + t).encode()).hexdigest() for t in texts]
        found, missing = {}, {}
        for key, text in zip(keys, texts):
            if key in found or key in missing:
                continue
            row = self.db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
            if row is not None:
                vector = np.frombuffer(row[0], dtype="<f4")
                if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise EmbeddingError("Corrupt embedding cache entry; use a new cache file")
                found[key] = vector
            else:
                missing[key] = text
        if missing:
            # Cache each bounded batch immediately, so interruptions don't lose an entire corpus.
            pairs = list(missing.items())
            for start in range(0, len(pairs), self.backend.batch_size):
                part = pairs[start:start + self.backend.batch_size]
                vectors = normalize_vectors(self.backend.encode([t for _, t in part]), len(part))
                for (key, _), vector in zip(part, vectors, strict=True):
                    found[key] = vector
                    self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                    (key, vector.astype("<f4").tobytes()))
                self.db.commit()
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        return np.stack([found[key] for key in keys])

    def snapshot(self):
        return dict(config=self.metadata, cache_hits=self.hits, cache_misses=self.misses,
                    backend_usage=dict(self.backend.stats))

    def close(self):
        self.db.close()
        self.backend.close()


class DenseRanker:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, query, items, limit):
        if not items:
            return []
        vectors = self.encoder.encode([x["text"] for x in items])
        query_vector = self.encoder.encode([query])[0]
        scores = vectors @ query_vector
        order = sorted(range(len(items)), key=lambda i: (-float(scores[i]), items[i]["id"]))[:limit]
        return [dict(items[i], score=round(float(scores[i]), 6)) for i in order]
