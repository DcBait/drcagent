"""Layout access for the agent: build test blocks, query shapes, and make edits.

A block is a row of real sky130_fd_sc_hd standard cells (read-only foundry IP)
plus top-level "routing" shapes, which are the only shapes the agent may edit.
"""
from __future__ import annotations

from pathlib import Path

import klayout.db as kdb

from .config import CELLS_DIR, LAYER_BY_NUMBER, LAYERS, snap

ID_KEY = 1  # GDS only stores properties with integer keys (PROPATTR)


def cell_gds(cell: str) -> Path:
    family = cell.replace("sky130_fd_sc_hd__", "").rsplit("_", 1)[0]
    return CELLS_DIR / family / f"{cell}.gds"


class Block:
    def __init__(self, layout: kdb.Layout, top: kdb.Cell):
        self.layout, self.top = layout, top
        self.dbu = layout.dbu
        self._next_id = 1 + max([self._id(s) or 0 for s in self._editable()], default=0)

    # construction / io ------------------------------------------------------
    @classmethod
    def from_cells(cls, cells: list[str], name: str = "block") -> "Block":
        layout = kdb.Layout()
        layout.dbu = 0.001
        top = layout.create_cell(name)
        x = 0.0
        for c in cells:
            tmp = kdb.Layout()
            tmp.read(str(cell_gds(c)))
            src = tmp.cell(c)
            dst = layout.create_cell(c) if not layout.has_cell(c) else layout.cell(c)
            if dst.is_empty():
                dst.copy_tree(src)
            width = src.dbbox().width()
            top.insert(kdb.DCellInstArray(dst.cell_index(), kdb.DTrans(x, 0.0)))
            x += width
        return cls(layout, top)

    @classmethod
    def load(cls, path: str | Path, top: str = "block") -> "Block":
        layout = kdb.Layout()
        layout.read(str(path))
        return cls(layout, layout.cell(top))

    def save(self, path: str | Path) -> Path:
        opts = kdb.SaveLayoutOptions()
        opts.write_context_info = False
        self.layout.write(str(path), opts)
        return Path(path)

    def copy(self) -> "Block":
        layout = kdb.Layout()
        layout.assign(self.layout)
        return Block(layout, layout.cell(self.top.name))

    # helpers -----------------------------------------------------------------
    def li(self, name: str) -> int:
        return self.layout.layer(*LAYERS[name])

    def _editable(self):
        for name in LAYERS:
            for s in self.top.shapes(self.li(name)).each():
                yield s

    def _id(self, shape) -> int | None:
        v = shape.property(ID_KEY)
        return int(v) if v is not None else None


    def _find(self, shape_id: int):
        if shape_id is None:
            raise KeyError("shape has no id; it is not editable")
        for s in self._editable():
            if self._id(s) == shape_id:
                return s
        raise KeyError(f"no editable shape with id {shape_id}")

    def _layer_name(self, layer_index: int) -> str:
        info = self.layout.get_info(layer_index)
        return LAYER_BY_NUMBER.get((info.layer, info.datatype), f"{info.layer}/{info.datatype}")

    @staticmethod
    def _box_dict(b: kdb.DBox) -> dict:
        return {"bbox": [round(b.left, 3), round(b.bottom, 3), round(b.right, 3), round(b.top, 3)],
                "width": round(b.width(), 3), "height": round(b.height(), 3),
                "area": round(b.width() * b.height(), 4)}

    def bbox(self) -> tuple[float, float, float, float]:
        b = self.top.dbbox()
        return b.left, b.bottom, b.right, b.top

    def region(self, layer: str, editable_only: bool = False) -> kdb.Region:
        if editable_only:
            return kdb.Region(self.top.shapes(self.li(layer)))
        return kdb.Region(self.top.begin_shapes_rec(self.li(layer)))

    # editing -----------------------------------------------------------------
    def add_box(self, layer: str, x1: float, y1: float, x2: float, y2: float) -> int:
        box = kdb.DBox(snap(x1), snap(y1), snap(x2), snap(y2))
        shape = self.top.shapes(self.li(layer)).insert(box)
        sid = self._next_id
        shape.set_property(ID_KEY, str(sid))
        self._next_id += 1
        return sid

    def shape(self, shape_id: int) -> dict:
        s = self._find(shape_id)
        d = {"id": shape_id, "layer": self._layer_name(s.layer), "editable": True}
        d.update(self._box_dict(s.dbbox()))
        return d

    def resize_shape(self, shape_id: int, left: float = 0, right: float = 0,
                     bottom: float = 0, top: float = 0) -> dict:
        """Move each edge outward (+) or inward (-) by the given amount in um."""
        s = self._find(shape_id)
        b = s.dbox if s.is_box() else s.dbbox()
        x1, y1 = snap(b.left - left), snap(b.bottom - bottom)
        x2, y2 = snap(b.right + right), snap(b.top + top)
        if x2 - x1 <= 0 or y2 - y1 <= 0:  # check before DBox, which silently re-orders corners
            raise ValueError("resize would remove the shape; shapes may not be deleted")
        nb = kdb.DBox(x1, y1, x2, y2)
        s.dbox = nb
        return self.shape(shape_id)

    def move_shape(self, shape_id: int, dx: float = 0, dy: float = 0) -> dict:
        s = self._find(shape_id)
        b = s.dbox if s.is_box() else s.dbbox()
        s.dbox = b.moved(snap(dx), snap(dy))
        return self.shape(shape_id)

    # querying ----------------------------------------------------------------
    def query(self, x: float, y: float, radius: float = 0.6, layers: list[str] | None = None,
              limit: int = 40) -> list[dict]:
        """Shapes whose bounding box comes within `radius` um of (x, y)."""
        win = kdb.DBox(x - radius, y - radius, x + radius, y + radius)
        out = []
        for name in layers or list(LAYERS):
            idx = self.li(name)
            for s in self.top.shapes(idx).each_overlapping(win):
                d = {"id": self._id(s), "layer": name, "editable": True, "owner": "top-level routing"}
                d.update(self._box_dict(s.dbbox()))
                out.append(d)
            it = self.top.begin_shapes_rec_overlapping(idx, win)
            while not it.at_end():
                if it.cell_index() != self.top.cell_index():
                    b = it.shape().dbbox().transformed(it.dtrans())
                    d = {"id": None, "layer": name, "editable": False,
                         "owner": f"foundry cell {self.layout.cell(it.cell_index()).name} (read-only)"}
                    d.update(self._box_dict(b))
                    out.append(d)
                it.next()
        out.sort(key=lambda d: (not d["editable"], d["layer"]))
        return out[:limit]

    def editable_shapes(self) -> list[dict]:
        return [self.shape(self._id(s)) for s in self._editable() if self._id(s) is not None]
