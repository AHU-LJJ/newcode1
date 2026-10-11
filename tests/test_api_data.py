import json

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from graphmem.api import APIError, ChatClient, Settings
from graphmem.data import load_examples
from graphmem.__main__ import summarize


def test_api_transport_and_secret_errors():
    settings = Settings("secret-test-key", "https://api.example/v1", "test-model")
    assert "secret-test-key" not in repr(settings)
    client = ChatClient(settings)
    client.http.close()
    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        assert json.loads(request.content)["tools"] == []
        return httpx.Response(401, text="secret-test-key must not appear in errors")
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(APIError) as error:
        client.complete([], [])
    assert "401" in str(error.value) and "secret-test-key" not in str(error.value)
    assert client.http_requests == 1
    client.close()


def test_parquet_sampling_and_label_boundary(tmp_path):
    rows = [dict(id=str(i), question="A question", answer="SECRET_GOLD",
                 type="bridge", context=dict(title=["Doc"], sentences=[["Evidence"]]),
                 supporting_facts=dict(title=["Doc"], sent_id=[0])) for i in range(5)]
    path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    first = load_examples(path, 3, 42)
    assert first == load_examples(path, 3, 42)
    assert len({e.public.id for e in first}) == 3
    assert "SECRET_GOLD" not in repr(first[0].public)
    with pytest.raises(ValueError):
        load_examples(path, 6, 42)


def test_api_failure_is_retained(question=None):
    from graphmem.data import Document, Question
    from graphmem.agent import run_agent
    from graphmem.metrics import score
    class FailingClient:
        def complete(self, messages, tools):
            raise APIError("API HTTP 401")
    q = Question("q", "Where?", (Document("Doc", ("London",)),))
    row = run_agent(q, "flat", FailingClient())
    assert row["status"] == "api_error" and not row["usage_complete"]
    row["metrics"] = score("", [], "London", [("Doc", 0)])
    result = summarize([row], ["flat"], 2)["modes"]["flat"]
    assert result["completed"] == 1 and result["successful"] == 0
    assert result["metrics"]["f1"] == 0
