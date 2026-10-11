import json

import pytest

from graphmem.agent import run_agent, schemas
from graphmem.data import Document, Question
from graphmem.graph import MemoryGraph
from graphmem.metrics import score


@pytest.fixture
def question():
    return Question("test", "Where was Ada's employer founded?", (
        Document("Ada", ("Ada worked at Engine Labs.",)),
        Document("Engine Labs", ("Engine Labs was founded in London.", "It later moved to Paris.")),
    ))


def test_evidence_gate_and_revision(question):
    graph = MemoryGraph(question)
    with pytest.raises(ValueError):
        graph.add_fact("Ada", "employer", "Engine Labs", ["s0:0"])
    graph.read_document("d0")
    fact = graph.add_fact("Ada", "employer", "Engine Labs", ["s0:0"])
    graph.read_document("d1")
    graph.revise_fact(fact["id"], "Engine Labs", "founded_in", "London", ["s1:0"], "Correct relation")
    assert (fact["id"], "supported_by", "s0:0") not in graph.edges
    assert (fact["id"], "supported_by", "s1:0") in graph.edges
    assert graph.events[1]["before"]["subject"] == "Ada"
    graph.retract_fact(fact["id"], "No longer useful")
    assert not any(n["id"] == fact["id"] for n in graph.search("London", graph=True)["nodes"])
    assert graph.nodes["s1:0"]["text"] == "Engine Labs was founded in London."
    with pytest.raises(ValueError):
        graph.expand(fact["id"])


def test_graph_traversal_and_evidence(question):
    graph = MemoryGraph(question)
    result = graph.expand("s0:0", hops=2)
    assert "s1:0" in {n["id"] for n in result["nodes"]}
    assert graph.supporting_pairs(["s1:0", "s1:0"]) == [["Engine Labs", 0]]
    with pytest.raises(ValueError, match="do not append"):
        graph.supporting_pairs(["s1:0:sent_id"])
    with pytest.raises(ValueError):
        graph.supporting_pairs(["f0"])
    assert len(graph.expand("d1", limit=1)["nodes"]) == 1
    assert graph.expand("d1", limit=1)["truncated"]


def test_metrics():
    m = score("The London", [["A", 0], ["B", 1]], "London", [("A", 0)])
    assert m["em"] == 1 and m["sp_f1"] == pytest.approx(2 / 3)
    assert m["joint_f1"] == pytest.approx(2 / 3)
    assert score("yes indeed", [], "yes", [])["f1"] == 0
    assert score("", [], "London", [("A", 0)])["joint_f1"] == 0


class ScriptedClient:
    def __init__(self, steps):
        self.steps = iter(steps)

    def complete(self, messages, tools):
        name, args = next(self.steps)
        return dict(choices=[dict(message=dict(role="assistant", content=None, tool_calls=[
            dict(id=f"call_{len(messages)}", type="function", function=dict(name=name, arguments=json.dumps(args)))
        ]))], usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15))


def test_tool_loop_and_no_gold(question):
    client = ScriptedClient([
        ("read_document", dict(document_id="d1")),
        ("add_fact", dict(subject="Engine Labs", relation="founded_in", object="London", evidence=["s1:0"])),
        ("finish", dict(answer="London", evidence=["s1:0"])),
    ])
    result = run_agent(question, "managed_graph", client, max_rounds=4)
    assert result["status"] == "ok" and result["api_calls"] == 3
    assert result["tokens"]["total_tokens"] == 45
    assert len(result["graph"]["events"]) == 1
    initial = json.loads(result["messages"][1]["content"])
    assert set(initial) == {"question", "documents"}


def test_disabled_tools_and_invalid_evidence(question):
    client = ScriptedClient([
        ("add_fact", dict(subject="X", relation="is", object="Y", evidence=["s0:0"])),
        ("finish", dict(answer="London", evidence=["s1:0"])),
    ])
    result = run_agent(question, "flat", client, max_rounds=2)
    assert result["status"] == "budget_exhausted" and result["tool_errors"] == 2
    assert not result["graph"]["events"]
    assert "add_fact" not in [s["function"]["name"] for s in schemas("static_graph")]


def test_episodes_are_isolated(question):
    first, second = MemoryGraph(question), MemoryGraph(question)
    first.read_document("d0")
    first.add_fact("Ada", "works_at", "Engine Labs", ["s0:0"])
    assert not second.seen and not second.events and "f0" not in second.nodes
