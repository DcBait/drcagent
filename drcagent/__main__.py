"""Command line.

    python -m drcagent build-bench                  # 40 verified cases (12 rule types)
    python -m drcagent build-bench --heldout        # 15 cases of rule types the baseline never saw
    python -m drcagent fix layout.gds --top block   # run the agent on one layout (needs ANTHROPIC_API_KEY)
    python -m drcagent eval --method baseline --bench benchmark
    python -m drcagent eval --method agent --bench benchmark_heldout --model claude-sonnet-5
    python -m drcagent models --provider gemini       # list models your GEMINI_API_KEY can use
    python -m drcagent eval --method agent --provider gemini --model <model> --bench benchmark_heldout
"""
from __future__ import annotations

import argparse
import json

from .inject import HELDOUT_RULES, build_benchmark
from .evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(prog="drcagent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-bench")
    b.add_argument("--out", default=None)
    b.add_argument("--n", type=int, default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--heldout", action="store_true")

    f = sub.add_parser("fix")
    f.add_argument("gds")
    f.add_argument("--top", default="block")
    f.add_argument("--model", default=None)
    f.add_argument("--provider", choices=["anthropic", "gemini", "openai"], default="anthropic")
    f.add_argument("--rpm", type=float, default=None, help="max requests per minute (free tiers)")
    f.add_argument("-o", "--output", default="fixed.gds")

    e = sub.add_parser("eval")
    e.add_argument("--method", choices=["baseline", "agent"], required=True)
    e.add_argument("--bench", default="benchmark")
    e.add_argument("--out", default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--provider", choices=["anthropic", "gemini", "openai"], default="anthropic")
    e.add_argument("--rpm", type=float, default=None, help="max requests per minute (free tiers)")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("-v", "--verbose", action="store_true")

    m = sub.add_parser("models")
    m.add_argument("--provider", choices=["anthropic", "gemini", "openai"], default="gemini")

    a = ap.parse_args()
    if a.cmd == "build-bench":
        if a.heldout:
            build_benchmark(a.out or "benchmark_heldout", a.n or 15, 1 if a.seed is None else a.seed,
                            rules=list(HELDOUT_RULES))
        else:
            build_benchmark(a.out or "benchmark", a.n or 40, 0 if a.seed is None else a.seed)
    elif a.cmd == "fix":
        from .agent import run_agent
        from .llm import make_provider
        from .tools import DRCSession
        s = DRCSession(a.gds, a.top)
        run = run_agent(s, verbose=True, provider=make_provider(a.provider, a.model, rpm=a.rpm))
        final = s.drc()
        s.save(a.output)
        print(json.dumps({"clean": final.clean, "remaining": final.counts(), "steps": run.steps,
                          "tokens": run.input_tokens + run.output_tokens, "report": s.report}, indent=1))
    elif a.cmd == "eval":
        split = a.bench.rstrip("/\\").replace("\\", "/").split("/")[-1]
        tag = a.method if a.method == "baseline" else f"agent_{a.provider}"
        out = a.out or f"results/{tag}_{split}"
        print(json.dumps(evaluate(a.method, a.bench, out, a.model, a.limit, a.verbose,
                                  provider=a.provider, rpm=a.rpm), indent=1))
    elif a.cmd == "models":
        from .llm import list_models
        print("\n".join(list_models(a.provider)))


if __name__ == "__main__":
    main()
