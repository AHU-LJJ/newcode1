import ast
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re

import httpx
import numpy as np
from nltk.stem import PorterStemmer
import pytest

from graphmem.data import load_locomo, resolve_evidence
from graphmem.embeddings import APIBGE, CachedEncoder, DenseRanker, EmbeddingError, DIMENSION
from graphmem.graph import MemoryGraph
from graphmem.metrics import score_locomo


class FakeBGE:
    metadata = {"backend": "fake", "dimension": DIMENSION}
    batch_size = 2
    stats = {}

    def __init__(self):
        self.encoded = []
        self.stats = {"texts": 0}

    def encode(self, texts):
        self.encoded.extend(texts)
        self.stats["texts"] += len(texts)
        vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0 if "London" in text else 1] = 1
        return vectors

    def close(self):
        pass


def test_cache_and_revised_graph_membership(tmp_path):
    from graphmem.data import Document, Question
    backend = FakeBGE()
    encoder = CachedEncoder(backend, tmp_path / "vectors.sqlite")
    v = encoder.encode(["London", "Paris", "London"])
    assert v.shape == (3, DIMENSION) and backend.encoded == ["London", "Paris"]
    assert np.array_equal(v[0], v[2])
    encoder.encode(["Paris", "London"])
    assert len(backend.encoded) == 2
    graph = MemoryGraph(Question("q", "Where?", (Document("Doc", ("London", "Paris")),)), DenseRanker(encoder))
    graph.read_document("d0")
    fact = graph.add_fact("Company", "based_in", "London", ["s0:0"])
    assert any(n["id"] == fact["id"] for n in graph.search("London", graph=True)["nodes"])
    graph.revise_fact(fact["id"], "Company", "based_in", "Paris", ["s0:1"], "Correct city")
    hit = next(n for n in graph.search("London", graph=True)["nodes"] if n["id"] == fact["id"])
    assert hit["object"] == "Paris" and hit["score"] == 0
    graph.retract_fact(fact["id"], "Discard")
    assert all(n["id"] != fact["id"] for n in graph.search("London", graph=True)["nodes"])
    encoder.close()
    restored = CachedEncoder(FakeBGE(), tmp_path / "vectors.sqlite")
    restored.encode(["London"])
    assert not restored.backend.encoded
    restored.close()


def test_embedding_api_order_dimension_and_redaction():
    api = APIBGE("secret-test", "https://api.example/v1", batch_size=2)
    api.client.close()
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content)["model"] == "BAAI/bge-m3"
        vectors = np.zeros((2, DIMENSION))
        vectors[0, 0], vectors[1, 1] = 1, 1
        return httpx.Response(200, json=dict(data=[dict(index=i, embedding=vectors[i].tolist()) for i in [1, 0]], usage=dict(total_tokens=8)))
    api.client = httpx.Client(transport=httpx.MockTransport(handler))
    v = api.encode(["A", "B"])
    assert v[0, 0] == 1 and v[1, 1] == 1 and api.stats["input_tokens"] == 8
    api.client.close()
    api.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret-test")))
    with pytest.raises(EmbeddingError) as error:
        api.encode(["A"])
    assert "secret-test" not in str(error.value)
    api.close()


@pytest.fixture
def locomo_path(tmp_path):
    data = [dict(sample_id="conv-test", event_summary="SECRET SUMMARY", observation="SECRET OBSERVATION",
        conversation=dict(session_1_date_time="8 May 2023", session_1=[
            dict(dia_id="D1:1", speaker="Ada", text="I paint.", blip_caption="a painting"),
            dict(dia_id="D1:2", speaker="Bob", text="I visited London yesterday.")]),
        qa=[dict(question="Where?", answer="SECRET GOLD", category=i, evidence=["D1:2"]) for i in [1, 2, 3, 4]] +
           [dict(question="Unknown?", adversarial_answer="SECRET DISTRACTOR", category=5, evidence=[])])]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(data))
    return path


def test_locomo_sampling_labels_and_ids(locomo_path):
    examples = load_locomo(locomo_path, 4, 42)
    assert [e.kind for e in examples] == ["1", "2", "3", "4"]
    assert examples == load_locomo(locomo_path, 4, 42)
    public = json.dumps(asdict(examples[0].public))
    assert "SECRET" not in public and "8 May 2023" in public and "a painting" in public
    graph = MemoryGraph(examples[0].public)
    assert graph.nodes["s0:1"]["source_id"] == "D1:2"
    assert any(relation == "speaker_of" for _, relation, _ in graph.edges)
    pairs, unknown = resolve_evidence(["D1:01; D1:2", "D", "D1:9"], examples[0].public.documents)
    assert len(pairs) == 2 and unknown == ("D", "D1:9")
    assert load_locomo(locomo_path, 1, 42, categories=(5,))[0].answer == "no information available"


def test_locomo_scores_match_original_evaluator(locomo_path):
    """Execute only the already-inspected scoring functions, without importing model/bert dependencies."""
    source = Path(__file__).resolve().parents[1] / "datasets/locomo/task_eval/evaluation.py"
    if not source.exists():
        pytest.skip("Original dataset evaluator not present")
    names = {"normalize_answer", "f1_score", "f1", "eval_question_answering"}
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"regex": re, "string": __import__("string"), "Counter": Counter, "ps": PorterStemmer(), "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
    examples = load_locomo(locomo_path, 5, 42, categories=(1, 2, 3, 4, 5))
    for category, prediction, answer in [(1, "yoga, paintings", "yoga, painting, dance"),
                                         (2, "May 7, 2023", "7 May 2023"),
                                         (3, "counseling", "counseling; psychology"),
                                         (4, "painting", "paintings"), (5, "not mentioned", "ignored")]:
        from dataclasses import replace
        example = replace(examples[category-1], answer=answer)
        native = dict(category=category, answer=answer, evidence=[], prediction=prediction)
        expected = scope["eval_question_answering"]([native], "prediction")[0][0]
        actual = score_locomo(dict(answer=prediction, supporting_facts=[]), example)["f1"]
        assert actual == pytest.approx(expected)


def test_resumed_cli_preserves_initial_embedding_usage(locomo_path, tmp_path, monkeypatch):
    from graphmem import __main__ as cli
    from graphmem.api import Settings
    import sys
    settings = Settings("fake-key", "https://api.example/v1", "fake-model")
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path: settings))
    monkeypatch.setattr(cli, "make_encoder", lambda args, config:
                        CachedEncoder(FakeBGE(), tmp_path / "cache.sqlite"))
    class Client:
        def __init__(self):
            self.http_requests = 0
        def complete(self, messages, tools):
            self.http_requests += 1
            if messages[-1]["role"] == "tool":
                name, arguments = "finish", dict(answer="London", evidence=["s0:1"])
            else:
                name, arguments = "search_sources", dict(query="London")
            return dict(choices=[dict(message=dict(content=None, tool_calls=[
                dict(id=f"call{self.http_requests}", function=dict(name=name, arguments=json.dumps(arguments)))
            ]))], usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        def close(self):
            pass
    monkeypatch.setattr(cli, "ChatClient", lambda *args: Client())
    out = tmp_path / "run"
    argv = ["graphmem", "run", "--dataset", "locomo", "--data", str(locomo_path),
            "--limit", "1", "--modes", "flat", "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    initial = (out / "embedding_usage.json").read_bytes()
    assert json.loads(initial)["backend_usage"]["texts"] > 0
    # Simulate an episode not committed at interruption; the content cache remains.
    next((out / "episodes").glob("*.json")).unlink()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    assert cli.main() == 0
    assert (out / "embedding_usage.json").read_bytes() == initial
    assert json.loads((out / "embedding_usage_resume.json").read_text())["backend_usage"]["texts"] == 0
