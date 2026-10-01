"""Paths and the sky130 layer map."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDK = Path(os.environ.get("DRCAGENT_PDK", ROOT / "pdk"))

DRC_DECK = PDK / "mpw_precheck/checks/tech-files/sky130A_mr.drc"
CELLS_DIR = PDK / "sc_hd/cells"
RULES_DIR = PDK / "skywater-pdk/docs/rules"

# sky130 layers used in this project: name -> (GDS layer, datatype)
LAYERS: dict[str, tuple[int, int]] = {
    "nwell": (64, 20),
    "diff": (65, 20),
    "tap": (65, 44),
    "poly": (66, 20),
    "licon1": (66, 44),
    "li1": (67, 20),
    "mcon": (67, 44),
    "met1": (68, 20),
    "via": (68, 44),
    "met2": (69, 20),
    "via2": (69, 44),
    "met3": (70, 20),
}
LAYER_BY_NUMBER = {v: k for k, v in LAYERS.items()}

GRID_UM = 0.005  # sky130 manufacturing grid


def snap(x_um: float) -> float:
    """Round a coordinate to the 5 nm manufacturing grid."""
    return round(round(x_um / GRID_UM) * GRID_UM, 3)
