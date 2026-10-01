"""Evaluate a fixer (the rule-based baseline or the LLM agent) on a benchmark.

Scoring never trusts the fixer: after it finishes, the layout is re-checked
with the real DRC deck. Metrics per case:
  clean           - DRC passes with zero violations
  edits, area     - how much the fixer changed, vs the minimal fix on record
  rule recall     - (agent) share of injected violations its report identifies
  false claims    - (agent) findings marked verified_clean on a layout that is not
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .baseline import run_baseline
from .inject import Case, load_benchmark
from .tools import DRCSession


def _near(loc, bbox, tol=0.3) -> bool:
    try:
        x, y = float(loc[0]), float(loc[1])
    except (TypeError, ValueError, IndexError):
        return False
    return bbox[0] - tol <= x <= bbox[2] + tol and bbox[1] - tol <= y <= bbox[3] + tol


def score_case(case: Case, session: DRCSession, extra: dict) -> dict:
    final = session.drc()
    cost = session.edit_cost()
    min_area = sum(i.min_fix_area for i in case.injected)
    row = {"case": case.name, "rules": [i.rule for i in case.injected], "clean": final.clean,
           "remaining": final.counts(), "drc_runs": session.drc_runs - 1, **cost,
           "min_fix_area_um2": round(min_area, 4)}
    if session.report is not None:
        findings = session.report.get("findings", [])
        found = sum(any(f.get("rule", "").lower() == i.rule and _near(f.get("location"), i.bbox)
                        for f in findings) for i in case.injected)
        row["rule_recall"] = round(found / len(case.injected), 3)
        row["false_clean_claims"] = sum(bool(f.get("verified_clean")) for f in findings) if not final.clean else 0
    row.update(extra)
    return row


def evaluate(method: str, bench_dir: str | Path, out_dir: str | Path, model: str | None = None,
             limit: int | None = None, verbose: bool = False, client=None,
             provider: str = "anthropic", rpm: float | None = None) -> dict:
    bench_dir, out_dir = Path(bench_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = load_benchmark(bench_dir)[:limit]
    for case in cases:
        session = DRCSession(bench_dir / case.gds)
        if method == "baseline":
            run_baseline(session)
            extra = {}
        elif method == "agent":
            from .agent import run_agent
            from .llm import make_provider
            run = run_agent(session, verbose=verbose, provider=make_provider(provider, model, client, rpm))
            model = run.model
            extra = {"provider": run.provider, "model": run.model, "steps": run.steps, "stopped": run.stopped, "input_tokens": run.input_tokens,
                     "output_tokens": run.output_tokens, "seconds": run.seconds}
            (out_dir / f"{case.name}.trace.json").write_text(json.dumps(
                {"run": asdict(run), "report": session.report, "edits": session.edits}, indent=1, default=str))
        else:
            raise ValueError(method)
        session.save(out_dir / f"{case.name}.fixed.gds")
        row = score_case(case, session, extra)
        rows.append(row)
        print(f"{case.name} {row['rules']} clean={row['clean']} edits={row['edits']}"
              + (f" recall={row.get('rule_recall')} tokens={row.get('input_tokens', 0) + row.get('output_tokens', 0)}"
                 if method == "agent" else ""))
    summary = summarize(rows, method, model)
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))
    return summary


def summarize(rows: list[dict], method: str, model: str | None) -> dict:
    n = len(rows)
    s = {"method": method, "model": model if method == "agent" else None, "cases": n,
         "violations": sum(len(r["rules"]) for r in rows),
         "clean_rate": round(sum(r["clean"] for r in rows) / n, 3) if n else 0,
         "mean_edits_per_violation": round(sum(r["edits"] for r in rows) / max(1, sum(len(r["rules"]) for r in rows)), 2),
         "area_changed_um2": round(sum(r["area_changed_um2"] for r in rows), 4),
         "min_fix_area_um2": round(sum(r["min_fix_area_um2"] for r in rows), 4)}
    if method == "agent" and n:
        s["rule_recall"] = round(statistics.mean(r.get("rule_recall", 0) for r in rows), 3)
        s["false_clean_claims"] = sum(r.get("false_clean_claims", 0) for r in rows)
        s["mean_steps"] = round(statistics.mean(r["steps"] for r in rows), 1)
        s["input_tokens"] = sum(r["input_tokens"] for r in rows)
        s["output_tokens"] = sum(r["output_tokens"] for r in rows)
        s["mean_seconds"] = round(statistics.mean(r["seconds"] for r in rows), 1)
    return s
