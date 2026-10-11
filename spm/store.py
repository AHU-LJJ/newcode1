import json
from pathlib import Path
import sqlite3
import numpy as np
from .models import Episode, PrefUnit, fingerprint


def stored_unit(value):
    data=json.loads(value)
    if 'target' in data or 'facet' in data:
        from .small_protocol import TypedUnit
        return TypedUnit.model_validate(data)
    return PrefUnit.model_validate(data)


class Store:
    def __init__(self, path, user_id):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,ord INTEGER,time TEXT,hash TEXT,json TEXT,done INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,status TEXT,json TEXT);
        CREATE TABLE IF NOT EXISTS vecs(kind TEXT,id TEXT,dim INTEGER,blob BLOB,PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS raw(id TEXT PRIMARY KEY,json TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,json TEXT);
        """)
        row = self.db.execute("SELECT value FROM meta WHERE key='user'").fetchone()
        if row and row[0] != user_id:
            raise ValueError("Cross-user store access refused")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('user',?)", (user_id,))
        self.db.commit()

    def close(self):
        self.db.close()

    def stage(self, episode):
        value = episode.model_dump(mode="json"); key = fingerprint(value)
        row = self.db.execute("SELECT hash,done FROM episodes WHERE id=?", (episode.id,)).fetchone()
        if row:
            if row[0] != key:
                raise ValueError("Immutable episode ID reused with different content")
            return bool(row[1])
        pending = self.db.execute('SELECT id FROM episodes WHERE done=0 LIMIT 1').fetchone()
        if pending:
            raise ValueError('Complete the staged episode before ingesting another episode')
        last = self.db.execute("SELECT MAX(ord) FROM episodes WHERE done=1").fetchone()[0]
        if last is not None and episode.order <= last:
            raise ValueError("Episodes must be ingested in increasing order")
        self.db.execute("INSERT INTO episodes(id,ord,time,hash,json) VALUES (?,?,?,?,?)",
                        (episode.id, episode.order, episode.time, key, json.dumps(value)))
        self.db.commit()
        return False

    def episodes(self):
        return [Episode.model_validate_json(r[0]) for r in self.db.execute("SELECT json FROM episodes ORDER BY ord,id")]

    def all_units(self, statuses=("active", "candidate", "pending")):
        if statuses is None:
            rows = self.db.execute("SELECT json FROM units ORDER BY id")
        else:
            rows = self.db.execute("SELECT json FROM units WHERE status IN (" + ",".join("?" for _ in statuses) + ") ORDER BY id", statuses)
        return [stored_unit(r[0]) for r in rows]

    def get(self, key):
        row = self.db.execute("SELECT json FROM units WHERE id=?", (key,)).fetchone()
        return stored_unit(row[0]) if row else None

    def put(self, unit, vec=None):
        self.db.execute("INSERT OR REPLACE INTO units VALUES (?,?,?)", (unit.id, unit.status, unit.model_dump_json()))
        if vec is not None:
            self.put_vector("unit", unit.id, vec)

    def put_vector(self, kind, key, vector):
        v = np.asarray(vector, dtype="<f4")
        if v.shape != (1024,) or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1) > 1e-3:
            raise ValueError("Store requires normalized BGE-M3 vectors")
        self.db.execute("INSERT OR REPLACE INTO vecs VALUES (?,?,?,?)", (kind, key, len(v), v.tobytes()))

    def matrix(self, units):
        vectors = []
        for unit in units:
            row = self.db.execute("SELECT dim,blob FROM vecs WHERE kind='unit' AND id=?", (unit.id,)).fetchone()
            if not row or row[0] != 1024:
                raise ValueError("Missing unit vector")
            vectors.append(np.frombuffer(row[1], dtype="<f4"))
        return np.stack(vectors) if vectors else np.empty((0,1024), dtype=np.float32)

    def raw_candidates(self, probes, k):
        rows = list(self.db.execute("SELECT raw.id,raw.json,vecs.blob FROM raw JOIN vecs ON raw.id=vecs.id AND vecs.kind='raw' ORDER BY raw.id"))
        if not rows:
            return []
        scores = (probes @ np.stack([np.frombuffer(r[2], dtype='<f4') for r in rows]).T).max(axis=0)
        return [dict(**json.loads(rows[i][1]), score=float(scores[i])) for i in np.argsort(-scores, kind='stable')[:k]]

    def event(self, **fields):
        self.db.execute("INSERT INTO events(json) VALUES (?)", (json.dumps(fields),))

    def snapshot(self):
        return dict(units=[u.model_dump(mode="json") for u in self.all_units(None)],
                    events=[json.loads(r[0]) for r in self.db.execute("SELECT json FROM events ORDER BY id")],
                    episodes=[dict(id=e.id,order=e.order,time=e.time) for e in self.episodes()])
