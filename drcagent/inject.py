"""Inject known DRC violations into clean layouts to build a labelled benchmark.

Each injector draws "routing" shapes in free space next to a row of clean
foundry cells, records the ground truth (rule, location, the minimal fix), and
the benchmark builder keeps a case only if the real DRC deck flags exactly the
intended rules - so every label is verified by the tool, not assumed.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import klayout.db as kdb

from .config import snap
from .drc import run_drc
from .layout import Block

CELL_LIBRARY = ["sky130_fd_sc_hd__inv_1", "sky130_fd_sc_hd__inv_2", "sky130_fd_sc_hd__inv_4",
                "sky130_fd_sc_hd__nand2_1", "sky130_fd_sc_hd__nand2_2", "sky130_fd_sc_hd__nor2_1",
                "sky130_fd_sc_hd__nor2_2", "sky130_fd_sc_hd__nand3_1", "sky130_fd_sc_hd__buf_1"]

# rule -> (kind, layer, limit). Limits are the sky130 values the deck enforces.
RULES = {
    "m1.1": ("width", "met1", 0.14), "m1.2": ("spacing", "met1", 0.14), "m1.6": ("area", "met1", 0.083),
    "m1.5": ("mcon_enclosure", "met1", 0.06),
    "li.1": ("width", "li1", 0.17), "li.3": ("spacing", "li1", 0.17), "li.6": ("area", "li1", 0.0561),
    "poly.1a": ("width", "poly", 0.15), "poly.2": ("spacing", "poly", 0.21),
    "m2.1": ("width", "met2", 0.14), "m2.2": ("spacing", "met2", 0.14), "m2.6": ("area", "met2", 0.0676),
}
# Held-out rule types: the rule-based baseline was NOT written for these.
HELDOUT_RULES = {
    "via.4a": ("via_m1_enclosure", "met1", 0.055),
    "m2.4": ("via_m2_enclosure", "met2", 0.055),
    "ct.2": ("mcon_spacing", "mcon", 0.19),
}
ALL_RULES = {**RULES, **HELDOUT_RULES}
BLOCKING = ["poly", "licon1", "li1", "mcon", "met1", "via", "met2", "diff", "tap", "nwell"]


@dataclass
class Injected:
    rule: str
    layer: str
    shape_ids: list[int]
    bbox: list[float]
    current: float          # the bad value that was drawn (um or um2)
    required: float         # the rule limit
    fix_hint: str           # the minimal correct fix, in words
    min_fix_area: float     # um2 of metal a minimal fix adds (used to score edit cost)


@dataclass
class Case:
    name: str
    cells: list[str]
    gds: str
    injected: list[Injected] = field(default_factory=list)


def _r(rng, lo, hi):
    return snap(rng.uniform(lo, hi))


def _footprint(rule: str, rng: random.Random) -> tuple[list[tuple[str, tuple]], dict]:
    """Shapes (relative to an origin) and ground-truth details for one violation."""
    kind, layer, limit = ALL_RULES[rule]
    if kind in ("via_m1_enclosure", "via_m2_enclosure"):
        # a 0.15 um via between met1 and met2; one side of one metal has too little overhang
        e = _r(rng, 0.02, 0.045)
        v, g = 0.15, 0.1
        bad = (-e, -g, v + g, v + g)
        good = (-g, -g, v + g, v + g)
        m1, m2 = (bad, good) if kind == "via_m1_enclosure" else (good, bad)
        shapes = [("met1", m1), ("via", (0, 0, v, v)), ("met2", m2)]
        h = v + 2 * g
        info = dict(current=e, required=limit, min_fix_area=round((limit - e) * h, 4),
                    fix_hint=f"{layer} overhangs the via by only {e} um on the left; extend that "
                             f"{layer} edge to at least {limit} um")
    elif kind == "mcon_spacing":
        bad = _r(rng, 0.10, 0.18)
        c = 0.17
        w = 2 * c + bad
        shapes = [("li1", (0, -0.06, w, c + 0.06)), ("mcon", (0, 0, c, c)), ("mcon", (c + bad, 0, w, c)),
                  ("met1", (-0.06, -0.06, w + 0.06, c + 0.06))]
        info = dict(current=bad, required=limit, min_fix_area=round((limit - bad) * (c + 0.12) * 2, 4),
                    fix_hint=f"the two mcons are {bad} um apart; move one to at least {limit} um and "
                             f"extend the li1 and met1 pads to keep covering it")
    elif kind == "width":
        bad = _r(rng, limit * 0.55, limit - 0.01)
        length = snap(max(1.0, 1.6 * 0.1 / bad))
        shapes = [(layer, (0, 0, bad, length))]
        info = dict(current=bad, required=limit, min_fix_area=round((limit - bad) * length, 4),
                    fix_hint=f"widen the {layer} wire from {bad} um to at least {limit} um")
    elif kind == "spacing":
        bad = _r(rng, limit * 0.45, limit - 0.01)
        w = snap(limit * 1.6)
        shapes = [(layer, (0, 0, w, 1.0)), (layer, (w + bad, 0, 2 * w + bad, 1.0))]
        info = dict(current=bad, required=limit, min_fix_area=0.0,
                    fix_hint=f"move the two {layer} shapes apart from {bad} um to at least {limit} um")
    elif kind == "area":
        w = snap({"met1": 0.2, "li1": 0.17, "met2": 0.2}[layer])
        h = _r(rng, w, limit / w * 0.95)
        shapes = [(layer, (0, 0, w, h))]
        info = dict(current=round(w * h, 4), required=limit, min_fix_area=round(limit - w * h, 4),
                    fix_hint=f"grow the {layer} island from {round(w * h, 4)} um2 to at least {limit} um2")
    elif kind == "mcon_enclosure":
        # m1.5 fails when BOTH edges at one corner of the mcon have < 0.06 um of met1 overhang.
        e = _r(rng, 0.03, 0.05)        # left overhang (bad)
        e2 = _r(rng, 0.03, 0.05)       # top overhang (bad); right and bottom are generous
        c, g = 0.17, 0.12              # mcon is 0.17 x 0.17
        shapes = [("li1", (0, -0.115, c, c + 0.115)), ("mcon", (0, 0, c, c)),
                  ("met1", (-e, -g, c + g, c + e2))]
        worst = min(e, e2)
        info = dict(current=worst, required=limit,
                    min_fix_area=round(min((limit - e) * (c + g + e2), (limit - e2) * (c + g + e)), 4),
                    fix_hint=f"met1 overhang of the mcon is {e} um (left) and {e2} um (top); extend "
                             f"either the left or the top met1 edge to at least {limit} um")
    else:
        raise ValueError(kind)
    return shapes, info


def _find_spot(block: Block, shapes, occupied: kdb.Region, rng: random.Random, clearance=0.7):
    x1, y1, x2, y2 = block.bbox()
    row = kdb.DBox(x1, y1, x2, y2)
    fx1 = min(s[1][0] for s in shapes); fy1 = min(s[1][1] for s in shapes)
    fx2 = max(s[1][2] for s in shapes); fy2 = max(s[1][3] for s in shapes)
    candidates = []
    for ox in [x1 - 1.5 + 0.1 * i for i in range(int((x2 - x1 + 3) / 0.1))]:
        for oy in [y1 - 3.5 + 0.1 * j for j in range(int((y2 - y1 + 7) / 0.1))]:
            fp = kdb.DBox(ox + fx1, oy + fy1, ox + fx2, oy + fy2).enlarged(clearance, clearance)
            if fp.overlaps(row.enlarged(0.3, 0.3)):
                continue
            candidates.append((ox, oy, fp))
    rng.shuffle(candidates)
    dbu = block.dbu
    for ox, oy, fp in candidates:
        probe = kdb.Region(fp.to_itype(dbu))
        if (occupied & probe).is_empty():
            return snap(ox), snap(oy), probe
    raise RuntimeError("no free space for injection")


def inject(block: Block, rules: list[str], rng: random.Random) -> list[Injected]:
    occupied = kdb.Region()
    for layer in BLOCKING:
        occupied += block.region(layer)
    out = []
    for rule in rules:
        shapes, info = _footprint(rule, rng)
        ox, oy, probe = _find_spot(block, shapes, occupied, rng)
        ids = [block.add_box(layer, ox + a, oy + b, ox + c, oy + d) for layer, (a, b, c, d) in shapes]
        occupied += probe
        xs = [ox + s[1][i] for s in shapes for i in (0, 2)]
        ys = [oy + s[1][i] for s in shapes for i in (1, 3)]
        out.append(Injected(rule=rule, layer=ALL_RULES[rule][1], shape_ids=ids,
                            bbox=[min(xs), min(ys), max(xs), max(ys)], **info))
    return out


def matches_ground_truth(result, injected: list[Injected], tol: float = 0.05) -> bool:
    """True if DRC flags exactly the injected rules, each inside its own injection."""
    if set(result.counts()) != {i.rule for i in injected}:
        return False
    for v in result.violations:
        hits = [i for i in injected if i.rule == v.rule and
                v.bbox[0] >= i.bbox[0] - tol and v.bbox[1] >= i.bbox[1] - tol and
                v.bbox[2] <= i.bbox[2] + tol and v.bbox[3] <= i.bbox[3] + tol]
        if not hits:
            return False
    return True


def build_benchmark(out_dir: str | Path, n_cases: int = 40, seed: int = 0,
                    max_violations: int = 3, rules: list[str] | None = None) -> list[Case]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cases, attempts, rejected = [], 0, 0
    rule_names = list(rules or RULES)
    while len(cases) < n_cases:
        attempts += 1
        cells = rng.sample(CELL_LIBRARY, 3)
        k = 1 if len(cases) < n_cases // 2 else rng.randint(2, max_violations)
        # cycle rules so every rule appears roughly equally often
        start = (len(cases) * 5) % len(rule_names)
        picked = list(dict.fromkeys(rule_names[(start + i * 7) % len(rule_names)] for i in range(k)))
        block = Block.from_cells(cells)
        injected = inject(block, picked, rng)
        name = f"case_{len(cases):03d}"
        gds = out_dir / f"{name}.gds"
        block.save(gds)
        result = run_drc(gds, "block")
        if not matches_ground_truth(result, injected):
            rejected += 1
            print(f"  rejected attempt {attempts} ({picked}): DRC found {result.counts()}")
            continue
        cases.append(Case(name=name, cells=cells, gds=gds.name, injected=injected))
        print(f"{name}: {picked} verified")
    manifest = {"seed": seed, "attempts": attempts, "rejected": rejected,
                "cases": [asdict(c) for c in cases]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return cases


def load_benchmark(bench_dir: str | Path) -> list[Case]:
    data = json.loads((Path(bench_dir) / "manifest.json").read_text())
    return [Case(name=c["name"], cells=c["cells"], gds=c["gds"],
                 injected=[Injected(**i) for i in c["injected"]]) for c in data["cases"]]
