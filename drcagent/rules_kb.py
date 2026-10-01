"""Retrieval over the official sky130 design-rule documentation (the RAG part).

Corpus: every periphery rule table in google/skywater-pdk docs/rules/periphery
(rule ID, description, flags, value) plus the description string of every
check in the DRC deck. Retrieval is BM25 over word tokens, with rule IDs kept
whole so a query like "m1.5" finds that exact rule.
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import DRC_DECK, RULES_DIR

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9_]+)*")
_DECK_OUT = re.compile(r'\.output\(\s*"([^"]+)",\s*"([^"]+)"', re.S)
_STOP = {"of", "the", "to", "a", "an", "and", "or", "for", "in", "by", "is", "be", "with", "on", "at", "um", "min"}


@dataclass
class RuleDoc:
    rule_id: str      # normalized ID, e.g. "m1.5"
    text: str
    value: str
    source: str       # file the text came from

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "text": self.text, "value": self.value, "source": self.source}


def normalize_id(raw: str) -> str:
    """'(li.1.-)' -> 'li.1', '(m1.1)' -> 'm1.1', 'ct.1_a' -> 'ct.1_a'."""
    s = raw.strip().strip("()").lower()
    s = re.sub(r"\.-$", "", s)
    return s


_SYNONYMS = [("local interconnect", "li"), ("li1", "li"), ("metal 1", "m1"), ("metal1", "m1"), ("met1", "m1"),
             ("metal 2", "m2"), ("metal2", "m2"), ("met2", "m2"), ("polysilicon", "poly"),
             ("spacing", "spacing space"), ("enclosure", "enclosure enclosed")]


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for a, b in _SYNONYMS:
        text = text.replace(a, b)
    toks = _TOKEN.findall(text)
    out = []
    for t in toks:
        if t in _STOP:
            continue
        out.append(t)
        if "." in t:  # also index the parts: "m1.5" -> "m1"
            out.append(t.split(".")[0])
    return out


def _load_docs(rules_dir: Path, deck: Path) -> list[RuleDoc]:
    docs = []
    for path in sorted((rules_dir / "periphery").glob("p[0-9]*.csv")):
        layer = re.sub(r"^p\d+-|_dotdash.*$", "", path.stem)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                desc = " ".join((row.get("Description") or "").replace("|", " ").split())
                value = f"{row.get('Value', '').strip()} {row.get('Unit', '').strip()}".strip()
                flags = (row.get("Flags") or "").strip()
                text = f"{row['Name']} [{layer} layer rules] {desc}"
                if value:
                    text += f" Value: {value}."
                if flags:
                    text += f" Flags: {flags}."
                docs.append(RuleDoc(normalize_id(row["Name"]), text, value, f"skywater-pdk docs/rules/periphery/{path.name}"))
    if deck.exists():
        for name, desc in _DECK_OUT.findall(deck.read_text(encoding="utf-8", errors="ignore")):
            docs.append(RuleDoc(normalize_id(name), f"DRC deck check {name}: {desc}", "", "sky130A_mr.drc"))
    return docs


class RuleKB:
    def __init__(self, docs: list[RuleDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.toks = [tokenize(d.text) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df = Counter(t for ts in self.toks for t in set(ts))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(ts) for ts in self.toks]
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [dict(self.docs[i].to_dict(), score=round(s, 2)) for s, i in scores[:k]]

    def lookup(self, rule_id: str) -> list[dict]:
        """Exact lookup by rule ID (also matches deck sub-checks like 'ct.1' -> 'ct.1_a')."""
        rid = normalize_id(rule_id)
        hits = [d for d in self.docs if d.rule_id == rid]
        if not hits:
            base = rid.split("_")[0]
            hits = [d for d in self.docs if d.rule_id == base or d.rule_id.startswith(base + "_")]
        return [d.to_dict() for d in hits]


@lru_cache(maxsize=1)
def default_kb() -> RuleKB:
    return RuleKB(_load_docs(RULES_DIR, DRC_DECK))
