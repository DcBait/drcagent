"""Rule-based baseline: the script a CAD engineer would write by hand.

It reads each marker, classifies the rule from the deck's description text
(width / spacing / area / enclosure), and applies a hard-coded minimal fix.
No LLM. It is the bar the agent has to beat to be worth its cost.
"""
from __future__ import annotations

import math
import re

from .config import GRID_UM
from .tools import DRCSession

PREFIX_LAYER = {"m1": "met1", "m2": "met2", "li": "li1", "poly": "poly"}


def _up(x: float) -> float:
    """Round up to the 5 nm grid (plus one grid step of margin against rounding)."""
    return round(math.ceil(x / GRID_UM - 1e-9) * GRID_UM, 3)


def _layer_for(rule: str) -> str | None:
    return PREFIX_LAYER.get(rule.split(".")[0])


def _shapes_at(session: DRCSession, v, layer: str) -> list[dict]:
    cx, cy = v.center
    r = max(v.bbox[2] - v.bbox[0], v.bbox[3] - v.bbox[1]) / 2 + 0.05
    return [s for s in session.block.query(cx, cy, r, [layer]) if s["editable"]]


def fix_one(session: DRCSession, v) -> str:
    desc = v.description.lower()
    layer = _layer_for(v.rule)
    if layer is None or v.rule_value is None:
        return "skipped: unknown layer or value"
    limit = v.rule_value
    if "enclosure" in desc and "mcon" in desc:
        m1 = _shapes_at(session, v, "met1")
        mcon = [s for s in session.block.query(*v.center, 0.2, ["mcon"])]
        if not m1 or not mcon:
            return "skipped: shapes not found"
        b, c = m1[0]["bbox"], mcon[0]["bbox"]
        over = {"left": c[0] - b[0], "bottom": c[1] - b[1], "right": b[2] - c[2], "top": b[3] - c[3]}
        h, w = b[3] - b[1], b[2] - b[0]
        cost = {k: (limit - o) * (h if k in ("left", "right") else w) for k, o in over.items() if o < limit}
        side = min(cost, key=cost.get)
        session.tool_resize_shape(m1[0]["id"], **{side: _up(limit - over[side])})
        return f"extended met1 {side} edge"
    shapes = _shapes_at(session, v, layer)
    if not shapes:
        return "skipped: no editable shape at marker"
    if "width" in desc:
        s = min(shapes, key=lambda s: min(s["width"], s["height"]))
        if s["width"] <= s["height"]:
            d = _up((limit - s["width"]) / 2)
            session.tool_resize_shape(s["id"], left=d, right=d)
        else:
            d = _up((limit - s["height"]) / 2)
            session.tool_resize_shape(s["id"], bottom=d, top=d)
        return "widened"
    if "spacing" in desc:
        if len(shapes) < 2:
            return "skipped: need two shapes"
        a, b = sorted(shapes, key=lambda s: (s["bbox"][0], s["bbox"][1]))[:2]
        gap_x = b["bbox"][0] - a["bbox"][2]
        gap_y = b["bbox"][1] - a["bbox"][3]
        if gap_x >= gap_y:
            session.tool_move_shape(b["id"], dx=_up(limit - gap_x))
        else:
            session.tool_move_shape(b["id"], dy=_up(limit - gap_y))
        return "moved apart"
    if "area" in desc:
        s = shapes[0]
        if s["width"] <= s["height"]:
            session.tool_resize_shape(s["id"], top=_up(limit / s["width"] - s["height"]))
        else:
            session.tool_resize_shape(s["id"], right=_up(limit / s["height"] - s["width"]))
        return "grown to min area"
    return "skipped: rule type not handled"


def run_baseline(session: DRCSession, max_rounds: int = 3) -> dict:
    log = []
    for _ in range(max_rounds):
        result = session.drc()
        if result.clean:
            break
        for v in result.violations:
            log.append({"rule": v.rule, "action": fix_one(session, v)})
    final = session.drc()
    return {"clean": final.clean, "remaining": final.counts(), "log": log}
