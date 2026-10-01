"""The tool surface the agent works through, shared by the LLM agent and the baseline.

A DRCSession wraps one layout. Every edit goes through it, is logged, and is
restricted to top-level routing shapes: foundry cells are read-only and shapes
can never be deleted, so a "fix" cannot cheat by removing the problem.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .drc import DRCResult, run_drc
from .layout import Block
from .rules_kb import RuleKB, default_kb


class DRCSession:
    def __init__(self, gds: str | Path, top: str = "block", kb: RuleKB | None = None):
        self.source = Path(gds)
        self.top = top
        self.block = Block.load(gds, top)
        self.kb = kb or default_kb()
        self.edits: list[dict] = []
        self.drc_runs = 0
        self.last: DRCResult | None = None
        self._tmp = tempfile.TemporaryDirectory()
        self.report: dict | None = None

    # actions ------------------------------------------------------------------
    def drc(self) -> DRCResult:
        path = Path(self._tmp.name) / f"work_{self.drc_runs}.gds"
        self.block.save(path)
        self.drc_runs += 1
        self.last = run_drc(path, self.top)
        return self.last

    def save(self, path: str | Path) -> Path:
        return self.block.save(path)

    # tools --------------------------------------------------------------------
    def tool_run_drc(self) -> dict:
        r = self.drc()
        by_rule = {}
        for v in r.violations:
            e = by_rule.setdefault(v.rule, {"count": 0, "description": v.description})
            e["count"] += 1
        return {"clean": r.clean, "total_violations": len(r.violations), "by_rule": by_rule,
                "rules_checked": r.rules_checked}

    def tool_list_violations(self, rule: str | None = None, limit: int = 20) -> dict:
        if self.last is None:
            self.drc()
        vs = [v for v in self.last.violations if rule is None or v.rule == rule]
        return {"violations": [dict(index=i, rule=v.rule, description=v.description, bbox=list(v.bbox),
                                    center=list(v.center), measured_um=v.measured_um, rule_value=v.rule_value,
                                    unit=v.unit, marker_kind=v.kind) for i, v in enumerate(vs[:limit])],
                "total": len(vs)}

    def tool_lookup_rule(self, rule_id: str) -> dict:
        hits = self.kb.lookup(rule_id)
        return {"rule_id": rule_id, "matches": hits[:6]} if hits else {"rule_id": rule_id, "matches": [],
                                                                        "note": "no exact match; try search_rules"}

    def tool_search_rules(self, query: str, k: int = 5) -> dict:
        return {"query": query, "results": self.kb.search(query, k)}

    def tool_inspect_layout(self, x: float, y: float, radius: float = 0.6, layers: list[str] | None = None) -> dict:
        return {"center": [x, y], "radius": radius, "shapes": self.block.query(x, y, radius, layers)}

    def tool_resize_shape(self, shape_id: int, left: float = 0, right: float = 0,
                          bottom: float = 0, top: float = 0) -> dict:
        before = self.block.shape(shape_id)
        after = self.block.resize_shape(shape_id, left, right, bottom, top)
        self.edits.append({"op": "resize", "shape_id": shape_id, "before": before, "after": after})
        return {"ok": True, "before": before, "after": after}

    def tool_move_shape(self, shape_id: int, dx: float = 0, dy: float = 0) -> dict:
        before = self.block.shape(shape_id)
        after = self.block.move_shape(shape_id, dx, dy)
        self.edits.append({"op": "move", "shape_id": shape_id, "before": before, "after": after})
        return {"ok": True, "before": before, "after": after}

    def tool_submit_report(self, findings: list[dict], summary: str) -> dict:
        self.report = {"findings": findings, "summary": summary}
        return {"ok": True}

    # dispatch -----------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), default=str)
        except Exception as e:  # tool errors go back to the model, not up the stack
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def edit_cost(self) -> dict:
        """Metal area added/removed and total move distance across all edits."""
        area, moved = 0.0, 0.0
        for e in self.edits:
            b, a = e["before"], e["after"]
            if e["op"] == "resize":
                area += abs(a["area"] - b["area"])
            else:
                moved += abs(a["bbox"][0] - b["bbox"][0]) + abs(a["bbox"][1] - b["bbox"][1])
        return {"edits": len(self.edits), "area_changed_um2": round(area, 4), "moved_um": round(moved, 3)}


TOOL_SCHEMAS = [
    {"name": "run_drc", "description": "Run the full sky130 DRC deck (181 checks) on the current layout and "
     "return violation counts per rule. Call it first, and again after edits to verify them.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_violations", "description": "List individual violations from the last DRC run: rule, the "
     "deck's description, marker bounding box and center (um), and for edge-pair markers the measured "
     "distance between the flagged edges.",
     "input_schema": {"type": "object", "properties": {
         "rule": {"type": "string", "description": "only this rule, e.g. 'm1.1'"},
         "limit": {"type": "integer", "default": 20}}}},
    {"name": "lookup_rule", "description": "Look up a rule ID in the official SkyWater sky130 design-rule "
     "documentation and the DRC deck. Returns the rule text and value.",
     "input_schema": {"type": "object", "properties": {"rule_id": {"type": "string"}}, "required": ["rule_id"]}},
    {"name": "search_rules", "description": "Free-text search over the sky130 design-rule documentation.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "k": {"type": "integer", "default": 5}},
                      "required": ["query"]}},
    {"name": "inspect_layout", "description": "List shapes near a point: layer, bbox [x1,y1,x2,y2] (um), "
     "width, height, area, and whether it is editable. Only editable shapes (top-level routing, with an id) "
     "can be changed; shapes inside foundry standard cells are read-only.",
     "input_schema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "radius": {"type": "number", "default": 0.6},
         "layers": {"type": "array", "items": {"type": "string"},
                    "description": "subset of: nwell diff tap poly licon1 li1 mcon met1 via met2 via2 met3"}},
         "required": ["x", "y"]}},
    {"name": "resize_shape", "description": "Move edges of an editable rectangle outward (positive) or inward "
     "(negative), in um. Coordinates snap to the 5 nm manufacturing grid. Shapes cannot be deleted.",
     "input_schema": {"type": "object", "properties": {
         "shape_id": {"type": "integer"}, "left": {"type": "number", "default": 0},
         "right": {"type": "number", "default": 0}, "bottom": {"type": "number", "default": 0},
         "top": {"type": "number", "default": 0}}, "required": ["shape_id"]}},
    {"name": "move_shape", "description": "Translate an editable rectangle by dx, dy um (snapped to 5 nm).",
     "input_schema": {"type": "object", "properties": {
         "shape_id": {"type": "integer"}, "dx": {"type": "number", "default": 0},
         "dy": {"type": "number", "default": 0}}, "required": ["shape_id"]}},
    {"name": "submit_report", "description": "Finish: submit one finding per violation you analysed. Call only "
     "after a final run_drc.",
     "input_schema": {"type": "object", "properties": {
         "findings": {"type": "array", "items": {"type": "object", "properties": {
             "rule": {"type": "string"},
             "location": {"type": "array", "items": {"type": "number"}, "description": "[x, y] um"},
             "explanation": {"type": "string", "description": "plain-English meaning of the rule and why "
                             "this geometry breaks it, citing the official rule text and value"},
             "fix_applied": {"type": "string"},
             "verified_clean": {"type": "boolean"}},
             "required": ["rule", "location", "explanation", "fix_applied", "verified_clean"]}},
         "summary": {"type": "string"}}, "required": ["findings", "summary"]}},
]
