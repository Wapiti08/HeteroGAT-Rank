from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


EType = Tuple[str, str, str]


def etype_to_str(etype: EType) -> str:
    return f"{etype[0]}|{etype[1]}|{etype[2]}"


def str_to_etype(s: str) -> EType:
    a, b, c = s.split("|", 2)
    return (a, b, c)


@dataclass(frozen=True)
class RarityStats:
    """Document frequency stats computed over *benign* graphs.

    We store df over graphs for pairs (etype, dst_key).
    """

    num_graphs: int
    df: Dict[Tuple[EType, str], int]

    def idf(self, *, etype: EType, dst_key: str) -> float:
        """Smoothed IDF: log((N+1)/(df+1))."""
        n = max(int(self.num_graphs), 0)
        d = int(self.df.get((etype, str(dst_key)), 0))
        return math.log((n + 1.0) / (d + 1.0))


def load_rarity_stats(path: str | Path) -> RarityStats:
    p = Path(path)
    obj = json.loads(p.read_text())
    n = int(obj.get("num_graphs", 0))
    raw_df = obj.get("df", {})
    df: Dict[Tuple[EType, str], int] = {}
    for k, v in raw_df.items():
        # key format: "<SRC|REL|DST>\t<dst_key>"
        et_s, dst = k.split("\t", 1)
        df[(str_to_etype(et_s), dst)] = int(v)
    return RarityStats(num_graphs=n, df=df)


def save_rarity_stats(path: str | Path, stats: RarityStats) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw_df = {f"{etype_to_str(et)}\t{dst}": int(v) for (et, dst), v in stats.df.items()}
    obj = {"num_graphs": int(stats.num_graphs), "df": raw_df}
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

