"""LLM agent: triages DRC violations, fixes them, and verifies its own fixes.

The loop is provider-agnostic (see llm.py): Claude via the Anthropic API, or Gemini /
OpenAI via an OpenAI-compatible API. The model only acts through the tools in
tools.py, and the final verdict always comes from re-running the real DRC deck -
never from the model's own claim that something is fixed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .llm import AnthropicProvider, make_provider
from .tools import TOOL_SCHEMAS, DRCSession

DEFAULT_MODEL = os.environ.get("DRCAGENT_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a physical-verification (CPV) assistant for the SkyWater sky130 130 nm process.
You are given a layout block: a row of foundry standard cells (read-only) plus top-level routing
shapes (editable). Your job:

1. Run DRC and list the violations.
2. For each violation, look up the official rule (lookup_rule) and inspect the geometry near the
   marker (inspect_layout) to identify the exact shape(s) at fault and by how much they miss the limit.
3. Fix it with the smallest edit that satisfies the rule: resize or move editable shapes only.
   Never touch foundry cells. Keep shapes on the 5 nm grid. Do not create new problems (check spacing
   to neighbours before growing or moving a shape).
4. Re-run DRC to verify. Iterate until clean or until you cannot make progress.
5. Call submit_report with one finding per violation: the rule, its location, a plain-English
   explanation that cites the official rule text and value, the fix you applied, and whether DRC
   confirmed it.

Be precise with numbers: compute the required change from the measured value and the rule value.
Rule wording matters - for example an "enclosure on 2 adjacent edges" rule is about a corner."""

USER_PROMPT = "Here is a layout block (top cell 'block'). Make it DRC-clean and report your findings."


@dataclass
class AgentRun:
    model: str
    provider: str = "anthropic"
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stopped: str = ""
    final_text: str = ""


def run_agent(session: DRCSession, client=None, model: str = DEFAULT_MODEL, max_steps: int = 40,
              verbose: bool = False, provider=None) -> AgentRun:
    """Run the agent. `provider` is an llm.py provider; by default Claude via `client`."""
    if provider is None:
        provider = AnthropicProvider(model, client=client)
    run = AgentRun(model=provider.model, provider=provider.name)
    provider.start(SYSTEM_PROMPT, TOOL_SCHEMAS, USER_PROMPT)
    t0 = time.perf_counter()
    for step in range(max_steps):
        turn = provider.step()
        run.steps = step + 1
        run.input_tokens += turn.input_tokens
        run.output_tokens += turn.output_tokens
        for text in turn.texts:
            run.final_text = text
            if verbose:
                print(f"[model] {text[:300]}")
        results = []
        for call in turn.tool_calls:
            out = session.call(call.name, call.args)
            run.tool_calls.append({"name": call.name, "input": call.args, "result_preview": out[:300]})
            if verbose:
                print(f"[tool] {call.name}({json.dumps(call.args)[:160]}) -> {out[:160]}")
            results.append((call, out))
        if session.report is not None:
            run.stopped = "submitted"
            break
        if not results:
            run.stopped = turn.stop_reason or "no_tool_call"
            break
        provider.add_tool_results(results)
    else:
        run.stopped = "max_steps"
    run.seconds = round(time.perf_counter() - t0, 1)
    return run


__all__ = ["run_agent", "make_provider", "AgentRun", "SYSTEM_PROMPT", "DEFAULT_MODEL"]
