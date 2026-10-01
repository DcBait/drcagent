import json
import math
import random
from types import SimpleNamespace as NS

import pytest

from drcagent.agent import run_agent
from drcagent.drc import parse_value, run_drc
from drcagent.inject import HELDOUT_RULES, RULES, inject, matches_ground_truth
from drcagent.layout import Block
from drcagent.rules_kb import default_kb, normalize_id
from drcagent.tools import DRCSession

CELLS = ["sky130_fd_sc_hd__inv_1", "sky130_fd_sc_hd__nand2_1", "sky130_fd_sc_hd__nor2_1"]


@pytest.fixture
def m1_width_case(tmp_path):
    block = Block.from_cells(CELLS)
    block.add_box("met1", 1.0, 5.0, 1.1, 6.2)    # 0.10 um wide: violates m1.1 (0.14)
    path = block.save(tmp_path / "case.gds")
    return path


# --- parsing / DRC -----------------------------------------------------------

def test_edge_pair_distance_parsed():
    kind, pts, d = parse_value("edge-pair: (0,0;0,1)/(0.12,1;0.12,0)")
    assert kind == "edge-pair" and d == pytest.approx(0.12)


def test_clean_cells_pass_drc(tmp_path):
    path = Block.from_cells(CELLS).save(tmp_path / "clean.gds")
    r = run_drc(path, "block")
    assert r.clean and r.rules_checked > 150


def test_injected_width_violation_is_measured(m1_width_case):
    r = run_drc(m1_width_case, "block")
    assert r.counts() == {"m1.1": 1}
    v = r.violations[0]
    assert v.measured_um == pytest.approx(0.10) and v.rule_value == pytest.approx(0.14)


@pytest.mark.parametrize("rule", ["m1.2", "li.6", "m1.5", "via.4a", "ct.2"])
def test_injection_is_verified_by_drc(rule, tmp_path):
    block = Block.from_cells(CELLS)
    injected = inject(block, [rule], random.Random(3))
    r = run_drc(block.save(tmp_path / "x.gds"), "block")
    assert matches_ground_truth(r, injected)


# --- layout tools -------------------------------------------------------------

def test_shape_ids_survive_gds_roundtrip(tmp_path):
    b = Block.from_cells(CELLS[:1])
    sid = b.add_box("met2", 0, 5, 0.3, 6)
    b2 = Block.load(b.save(tmp_path / "rt.gds"))
    assert [s["id"] for s in b2.editable_shapes()] == [sid]


def test_foundry_shapes_are_read_only(m1_width_case):
    s = DRCSession(m1_width_case)
    near = s.block.query(0.2, 0.0, 0.5)
    ro = [x for x in near if not x["editable"]]
    assert ro and all(x["id"] is None for x in ro)
    assert "error" in json.loads(s.call("resize_shape", {"shape_id": 999, "left": 0.1}))


def test_shapes_cannot_be_deleted(m1_width_case):
    s = DRCSession(m1_width_case)
    sid = s.block.editable_shapes()[0]["id"]
    out = json.loads(s.call("resize_shape", {"shape_id": sid, "left": -0.06, "right": -0.06}))
    assert "error" in out


def test_edits_snap_to_grid(m1_width_case):
    s = DRCSession(m1_width_case)
    sid = s.block.editable_shapes()[0]["id"]
    after = s.tool_resize_shape(sid, right=0.0331)["after"]
    assert math.isclose(round(after["bbox"][2] / 0.005) * 0.005, after["bbox"][2], abs_tol=1e-9)


# --- rule knowledge base ----------------------------------------------------------

def test_rule_ids_normalized():
    assert normalize_id("(li.1.-)") == "li.1" and normalize_id("(m1.5)") == "m1.5"


@pytest.mark.parametrize("rule", list(RULES) + list(HELDOUT_RULES))
def test_every_benchmark_rule_is_in_the_docs(rule):
    assert default_kb().lookup(rule), rule


@pytest.mark.parametrize("query,expected", [("minimum area of local interconnect", "li.6"),
                                            ("metal1 width", "m1.1"), ("met2 spacing", "m2.2")])
def test_search_finds_rule(query, expected):
    assert expected in [r["rule_id"] for r in default_kb().search(query, 3)]


# --- agent loop with a scripted fake model ---------------------------------------------

class ScriptedModel:
    """Stands in for the LLM: follows a fixed triage policy using only tool results."""

    def __init__(self):
        self.messages = self
        self.step = 0

    def create(self, messages, **kwargs):
        last = messages[-1]["content"]
        result = json.loads(last[-1]["content"]) if isinstance(last, list) else {}
        self.step += 1
        plan = {1: ("run_drc", {}), 2: ("list_violations", {})}
        if self.step in plan:
            name, args = plan[self.step]
        elif self.step == 3:
            v = result["violations"][0]
            self.v = v
            name, args = "inspect_layout", {"x": v["center"][0], "y": v["center"][1], "radius": 0.3}
        elif self.step == 4:
            shape = next(s for s in result["shapes"] if s["editable"])
            d = (self.v["rule_value"] - shape["width"]) / 2 + 0.005
            name, args = "resize_shape", {"shape_id": shape["id"], "left": d, "right": d}
        elif self.step == 5:
            name, args = "run_drc", {}
        else:
            name, args = "submit_report", {"summary": "fixed", "findings": [{
                "rule": self.v["rule"], "location": self.v["center"], "explanation": "too narrow",
                "fix_applied": "widened", "verified_clean": result.get("clean", False)}]}
        return NS(content=[NS(type="tool_use", id=f"t{self.step}", name=name, input=args)],
                  usage=NS(input_tokens=100, output_tokens=20), stop_reason="tool_use")


def test_agent_loop_end_to_end(m1_width_case):
    s = DRCSession(m1_width_case)
    run = run_agent(s, client=ScriptedModel(), model="fake")
    assert run.stopped == "submitted" and run.steps == 6
    assert s.drc().clean
    assert s.report["findings"][0]["verified_clean"] is True
    assert run.input_tokens == 600


def test_tools_can_solve_heldout_rule(tmp_path):
    """The held-out ct.2 case is fixable with the agent's tools (move + resize)."""
    block = Block.from_cells(CELLS)
    inj = inject(block, ["ct.2"], random.Random(4))[0]
    s = DRCSession(block.save(tmp_path / "ct.gds"))
    li, mcon_a, mcon_b, m1 = inj.shape_ids
    dx = round(inj.required - inj.current + 0.005, 3)
    s.tool_move_shape(mcon_b, dx=dx)
    s.tool_resize_shape(li, right=dx)
    s.tool_resize_shape(m1, right=dx)
    assert s.drc().clean


def test_requests_are_valid_for_the_real_sdk(m1_width_case):
    """Send the agent's requests through the real Anthropic SDK against a mock server."""
    anthropic = pytest.importorskip("anthropic")
    try:  # newer SDKs are built on httpx2
        import httpx2 as httpx
    except ImportError:
        import httpx

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "run_drc", "input": {}}]
        else:
            content = [{"type": "tool_use", "id": "toolu_2", "name": "submit_report",
                        "input": {"summary": "s", "findings": []}}]
        return httpx.Response(200, json={
            "id": f"msg_{len(seen)}", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 50, "output_tokens": 10}})

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    run = run_agent(DRCSession(m1_width_case), client=client, model="claude-sonnet-5")
    assert run.stopped == "submitted" and len(seen) == 2
    first, second = seen
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert {t["name"] for t in first["tools"]} >= {"run_drc", "inspect_layout", "submit_report"}
    tool_result = second["messages"][-1]["content"][-1]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])["by_rule"]["m1.1"]["count"] == 1


def test_openai_compatible_provider_end_to_end(m1_width_case):
    """Run the full agent loop through the real OpenAI SDK against a mock OpenAI-compatible endpoint."""
    openai = pytest.importorskip("openai")
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from drcagent.llm import GEMINI_BASE_URL, OpenAICompatProvider

    seen, state = [], {}

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        msgs = body["messages"]
        last = json.loads(msgs[-1]["content"]) if msgs[-1]["role"] == "tool" else {}
        n = len(seen)
        if n == 1:
            name, args = "run_drc", {}
        elif n == 2:
            name, args = "list_violations", {}
        elif n == 3:
            state["v"] = v = last["violations"][0]
            name, args = "inspect_layout", {"x": v["center"][0], "y": v["center"][1], "radius": 0.3}
        elif n == 4:
            shape = next(s for s in last["shapes"] if s["editable"])
            d = (state["v"]["rule_value"] - shape["width"]) / 2 + 0.005
            name, args = "resize_shape", {"shape_id": shape["id"], "left": d, "right": d}
        elif n == 5:
            name, args = "run_drc", {}
        else:
            name, args = "submit_report", {"summary": "ok", "findings": [{
                "rule": "m1.1", "location": state["v"]["center"], "explanation": "too narrow",
                "fix_applied": "widened", "verified_clean": last["clean"]}]}
        return httpx.Response(200, json={
            "id": f"c{n}", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"call{n}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95}})

    client = openai.OpenAI(base_url=GEMINI_BASE_URL, api_key="test",
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    s = DRCSession(m1_width_case)
    run = run_agent(s, provider=OpenAICompatProvider("gemini-test", client=client, name="gemini"))
    assert run.stopped == "submitted" and run.provider == "gemini" and run.steps == 6
    assert s.drc().clean and s.report["findings"][0]["verified_clean"] is True
    tools = {t["function"]["name"]: t["function"] for t in seen[0]["tools"]}
    assert "parameters" not in tools["run_drc"]                       # parameterless tool
    assert "default" not in json.dumps(seen[0]["tools"])              # schema keys Gemini rejects
    assert seen[0]["messages"][0]["role"] == "system"
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call1",
                                       "content": seen[1]["messages"][-1]["content"]}



def test_gemini_native_provider_end_to_end(m1_width_case):
    """Run the agent through the real google-genai SDK against a mock Gemini server.

    This is the path that works with the new "AQ." Gemini API keys.
    """
    genai = pytest.importorskip("google.genai")
    from google.genai import types
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    from drcagent.llm import GeminiProvider

    seen, state = [], {}

    def handler(request):
        body = json.loads(request.content)
        seen.append((request, body))
        contents = body["contents"]
        last_parts = contents[-1]["parts"]
        last = last_parts[-1].get("functionResponse", {}).get("response", {}).get("result", {})
        n = len(seen)
        if n == 1:
            name, args = "run_drc", {}
        elif n == 2:
            name, args = "list_violations", {}
        elif n == 3:
            state["v"] = v = last["violations"][0]
            name, args = "inspect_layout", {"x": v["center"][0], "y": v["center"][1], "radius": 0.3}
        elif n == 4:
            shape = next(s for s in last["shapes"] if s["editable"])
            d = (state["v"]["rule_value"] - shape["width"]) / 2 + 0.005
            name, args = "resize_shape", {"shape_id": shape["id"], "left": d, "right": d}
        elif n == 5:
            name, args = "run_drc", {}
        else:
            name, args = "submit_report", {"summary": "ok", "findings": [{
                "rule": "m1.1", "location": state["v"]["center"], "explanation": "too narrow",
                "fix_applied": "widened", "verified_clean": last["clean"]}]}
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": name, "args": args}, "thoughtSignature": "c2ln"}]},
                "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 70, "candidatesTokenCount": 12, "thoughtsTokenCount": 3}})

    client = genai.Client(api_key="AQ.test-key", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))
    s = DRCSession(m1_width_case)
    run = run_agent(s, provider=GeminiProvider("gemini-test", client=client))
    assert run.stopped == "submitted" and run.provider == "gemini" and run.steps == 6
    assert s.drc().clean and s.report["findings"][0]["verified_clean"] is True
    assert run.input_tokens == 6 * 70 and run.output_tokens == 6 * 15
    request, body = seen[0]
    assert request.headers.get("x-goog-api-key") == "AQ.test-key"          # one credential, no Bearer
    assert "authorization" not in {k.lower() for k in request.headers}
    decls = {d["name"]: d for d in body["tools"][0]["functionDeclarations"]}
    assert "parametersJsonSchema" not in decls["run_drc"]
    assert "default" not in json.dumps(body["tools"])
    # the model's turn is sent back unchanged (thought signature kept) with our function response
    _, second = seen[1]
    assert second["contents"][1]["parts"][0]["thoughtSignature"] == "c2ln"
    assert second["contents"][2]["parts"][0]["functionResponse"]["name"] == "run_drc"
