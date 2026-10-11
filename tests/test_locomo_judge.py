import json
import math

import httpx
import pytest

from graphmem.api import Settings
from graphmem.data import load_locomo
from graphmem import evaluate_locomo as evaluation
from graphmem.metrics import locomo_bleu1, score_locomo


def test_bleu1_clipping_brevity_and_reference_conventions():
    assert locomo_bleu1("red", "red blue", 4) == pytest.approx(math.exp(-1))
    assert locomo_bleu1("red red red", "red blue", 4) == pytest.approx(1 / 3)
    assert locomo_bleu1("The blue and red!", "red blue", 4) == 1
    assert locomo_bleu1("yes", "Yes; explanatory rationale", 3) == 1
    assert locomo_bleu1("", "yes", 3) == 0
    assert locomo_bleu1("green", "red blue", 4) == 0


def test_judge_validation_and_blinding():
    for value in [dict(score=True, reason="yes"), dict(score=2, reason="yes"),
                  dict(score=0.5, reason="partial"), dict(score=0, reason=""),
                  dict(score=1, reason="yes", extra="x")]:
        with pytest.raises(ValueError):
            evaluation.parse_grade(json.dumps(value))
    value = evaluation.judge_input(dict(question="Where?", reference="Paris", answer="London",
                                        mode="managed_graph", category=4, trace="secret"))
    assert set(value) == {"question", "reference_answer", "generated_answer"}
    assert "managed_graph" not in json.dumps(value)


def test_memory_r1_label_mapping_and_exact_template_transport():
    assert evaluation.parse_grade('{"label":"CORRECT"}', "memory-r1") == dict(
        label="CORRECT", score=1, reason=None)
    assert evaluation.parse_grade('{"label":"WRONG","explanation":"Different topic."}',
                                  "memory-r1")["score"] == 0
    for text in ['{}', '{"label":"correct"}', '{"label":true}', '{"label":"UNKNOWN"}',
                 '{"score":1,"reason":"yes"}', '{"label":"WRONG","explanation":42}']:
        with pytest.raises(ValueError):
            evaluation.parse_grade(text, "memory-r1")
    value = dict(question="Where was {someone}?", reference_answer="Paris", generated_answer="Paris, France")
    client = evaluation.JudgeClient(Settings("fake-key", "https://api.example/v1", "judge"), rubric="memory-r1")
    client.http.close()
    def handle(request):
        payload = json.loads(request.content)
        assert payload["messages"] == [dict(role="user", content=evaluation.MEMORY_R1_PROMPT.format(
            question=value["question"], gold_answer=value["reference_answer"], generated_answer=value["generated_answer"]))]
        assert "require all requested reference items" not in payload["messages"][0]["content"]
        return httpx.Response(200, json=dict(choices=[dict(finish_reason="stop",
            message=dict(content='{"label":"CORRECT"}'))], usage=dict(prompt_tokens=30,completion_tokens=5,total_tokens=35)))
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    result = client.grade(value)
    assert result["status"] == "ok" and result["score"] == 1 and result["label"] == "CORRECT"
    assert result["reason"] is None
    client.close()


def test_judge_transport_schema_retry_usage_and_redaction(monkeypatch):
    monkeypatch.setattr(evaluation.time, "sleep", lambda seconds: None)
    client = evaluation.JudgeClient(Settings("secret-key", "https://api.example/v1", "judge"))
    client.http.close()
    responses = [httpx.Response(200, json=dict(model="judge", choices=[dict(finish_reason="stop",
                 message=dict(content='{"score": 0.5, "reason": "partial"}'))],
                 usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30))),
                 httpx.Response(200, json=dict(model="judge", choices=[dict(finish_reason="stop",
                 message=dict(content='{"score": 1, "reason": "Equivalent."}'))],
                 usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30)))]
    def handle(request):
        payload = json.loads(request.content)
        assert "tools" not in payload and payload["enable_thinking"] is False
        assert payload["model"] == "judge" and payload["response_format"] == {"type": "json_object"}
        return responses.pop(0)
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    result = client.grade(dict(question="Where?", reference_answer="Paris", generated_answer="Paris"))
    assert result["status"] == "ok" and result["score"] == 1
    assert len(result["attempts"]) == 2 and sum(a["usage"]["total_tokens"] for a in result["attempts"]) == 60
    client.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401, text="secret-key")))
    failed = client.grade({})
    assert failed["status"] == "error" and failed["score"] is None
    assert "secret-key" not in json.dumps(failed) and len(failed["attempts"]) == 1
    client.close()


def test_incomplete_judge_coverage_is_not_a_zero_or_partial_average():
    rows = [dict(mode="flat", category=4, qa_status="ok", judge_keys=[k],
                 metrics=dict(f1=1, b1=1)) for k in ("a", "b")]
    scores = dict(a=dict(status="ok", score=1, attempts=[]),
                  b=dict(status="error", score=None, attempts=[]))
    result = evaluation.summarize(rows, ["flat"], scores, 1)
    assert result["modes"]["flat"]["j"] is None
    assert result["modes"]["flat"]["judge_scored"] == 1
    assert rows[1]["metrics"]["j"] is None


@pytest.fixture
def saved_run(tmp_path):
    data = tmp_path / "locomo.json"
    data.write_text(json.dumps([dict(sample_id="conv-test", conversation=dict(
        session_1_date_time="May 2023", session_1=[dict(dia_id="D1:1", speaker="Ada", text="Paris")]),
        qa=[dict(category=4, answer="Paris", question="Where?", evidence=["D1:1"])])]))
    example = load_locomo(data, 1, 42, categories=(4,))[0]
    run = tmp_path / "run"
    config = dict(dataset="locomo", data_sha256=evaluation.file_hash(data), limit=1, seed=42,
                  offset=0, categories=[4], sample_ids=[example.public.id], modes=["flat", "managed_graph"])
    evaluation.write_json(run / "manifest.json", dict(config=config))
    evaluation.write_json(run / "gold.json", [dict(_id=example.public.id, answer=example.answer,
        type=example.kind, conversation_id=example.public.conversation_id, qa_index=example.qa_index)])
    prediction = dict(answer="Paris", supporting_facts=[[example.public.documents[0].title, 0]])
    for mode in config["modes"]:
        evaluation.write_json(run / "episodes" / f"{mode}.json", dict(id=example.public.id, mode=mode,
            status="ok", prediction=prediction, metrics=score_locomo(prediction, example)))
    return run, data


def test_dedup_resume_and_source_integrity(saved_run, monkeypatch):
    run, data = saved_run
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path:
        Settings("fake-key", "https://api.example/v1", "qa-model")))
    calls = []
    class FakeJudge:
        def __init__(self, settings, *args):
            assert settings.model == evaluation.DEFAULT_MODEL
        def grade(self, value):
            calls.append(value)
            return dict(status="ok", score=1, reason="Correct.", attempts=[dict(usage_complete=True,
                         usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30))])
        def close(self):
            pass
    monkeypatch.setattr(evaluation, "JudgeClient", FakeJudge)
    args = ["--run", str(run), "--data", str(data), "--workers", "1"]
    assert evaluation.main(args) == 0 and len(calls) == 1
    assert evaluation.main(args + ["--resume"]) == 0 and len(calls) == 1
    summary = json.loads((run / "evaluations/deepseek_v32_memory_r1/summary.json").read_text())
    assert summary["unique_judgments"] == 1 and summary["tokens"]["total_tokens"] == 30
    assert summary["protocol"] == evaluation.RUBRICS["memory-r1"][0]
    assert all(v["j"] == 1 for v in summary["modes"].values())
    path = run / "episodes/flat.json"
    value = json.loads(path.read_text()); value["prediction"]["answer"] = "Berlin"
    evaluation.write_json(path, value)
    with pytest.raises(ValueError, match="F1"):
        evaluation.main(args + ["--resume"])


def test_rubric_caches_cannot_mix(saved_run, monkeypatch):
    run, data = saved_run
    output = run / "evaluation"
    args = ["--run", str(run), "--data", str(data), "--no-judge", "--output", str(output)]
    assert evaluation.main(args) == 0
    with pytest.raises(ValueError, match="exact same evaluation config"):
        evaluation.main(args + ["--judge-rubric", "strict", "--resume"])
