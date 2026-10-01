"""Run the sky130 KLayout DRC deck and parse its report database (.lyrdb).

The deck is the open-source sky130A_mr.drc from efabless' mpw_precheck: the
same rules used to check real chip submissions for SkyWater 130 nm shuttles.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DRC_DECK

_NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
_EDGE_PAIR = re.compile(rf"edge-pair:\s*\(({_NUM}),({_NUM});({_NUM}),({_NUM})\)[/|]\(({_NUM}),({_NUM});({_NUM}),({_NUM})\)")
_POINTS = re.compile(rf"\(?({_NUM}),({_NUM})\)?")
_RULE_VALUE = re.compile(rf":\s*({_NUM})\s*um(²|2)?", re.IGNORECASE)


@dataclass
class Violation:
    rule: str
    description: str
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in um
    kind: str                                  # "edge-pair" or "polygon"
    measured_um: float | None = None           # distance between the flagged edges
    rule_value: float | None = None            # limit stated by the rule
    unit: str = "um"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["center"] = self.center
        return d


@dataclass
class DRCResult:
    violations: list[Violation] = field(default_factory=list)
    rules_checked: int = 0
    seconds: float = 0.0

    @property
    def clean(self) -> bool:
        return not self.violations

    def counts(self) -> dict[str, int]:
        return dict(Counter(v.rule for v in self.violations))


def _seg_dist(p, q, a, b) -> float:
    """Minimum distance between segments pq and ab (2-D)."""
    def pt_seg(pt, s0, s1):
        (px, py), (x0, y0), (x1, y1) = pt, s0, s1
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2))
        return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
    return min(pt_seg(p, a, b), pt_seg(q, a, b), pt_seg(a, p, q), pt_seg(b, p, q))


def parse_value(text: str):
    """Parse one lyrdb marker value into (kind, points, measured distance)."""
    m = _EDGE_PAIR.search(text)
    if m:
        x = [float(v) for v in m.groups()]
        p, q, a, b = (x[0], x[1]), (x[2], x[3]), (x[4], x[5]), (x[6], x[7])
        return "edge-pair", [p, q, a, b], round(_seg_dist(p, q, a, b), 4)
    pts = [(float(a), float(b)) for a, b in _POINTS.findall(text)]
    return "polygon", pts, None


def parse_lyrdb(path: str | Path) -> DRCResult:
    root = ET.parse(path).getroot()
    descriptions = {}
    for c in root.find("categories").iter("category"):
        descriptions[c.findtext("name").strip("'")] = (c.findtext("description") or "").strip()
    result = DRCResult(rules_checked=len(descriptions))
    for item in root.iter("item"):
        rule = item.findtext("category").strip("'")
        desc = descriptions.get(rule, "")
        rv = _RULE_VALUE.search(desc)
        for value in item.iter("value"):
            kind, pts, measured = parse_value(value.text or "")
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            result.violations.append(Violation(
                rule=rule, description=desc,
                bbox=(min(xs), min(ys), max(xs), max(ys)), kind=kind, measured_um=measured,
                rule_value=float(rv.group(1)) if rv else None,
                unit="um2" if rv and rv.group(2) else "um"))
    return result


def run_drc(gds: str | Path, top_cell: str, deck: str | Path = DRC_DECK,
            timeout: int = 600) -> DRCResult:
    """Run the full sky130 front-end + back-end deck on one GDS file."""
    import time

    gds, deck = Path(gds).resolve(), Path(deck).resolve()
    if not deck.exists():
        raise FileNotFoundError(f"DRC deck not found at {deck}; run scripts/fetch_pdk.sh first")
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.lyrdb"
        cmd = [os.environ.get("KLAYOUT_BIN", "klayout"), "-b", "-r", str(deck), "-rd", f"input={gds}", "-rd", f"top_cell={top_cell}",
               "-rd", f"report={report}", "-rd", "feol=true", "-rd", "beol=true"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if not report.exists():
            raise RuntimeError(f"KLayout DRC produced no report:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        result = parse_lyrdb(report)
        result.seconds = round(time.perf_counter() - t0, 2)
    return result
