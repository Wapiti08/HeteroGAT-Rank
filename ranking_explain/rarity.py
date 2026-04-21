from __future__ import annotations

import ipaddress
import json
import math
import re
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
    normalize: str = "none"

    def idf(self, *, etype: EType, dst_key: str) -> float:
        """Smoothed IDF: log((N+1)/(df+1))."""
        n = max(int(self.num_graphs), 0)
        d = int(self.df.get((etype, str(dst_key)), 0))
        return math.log((n + 1.0) / (d + 1.0))


def normalize_dst_key(*, scheme: str, etype: EType, dst_type: str, dst_key: str) -> str:
    """Normalize destination keys to reduce long-tail explosions.

    This is primarily useful for OSPTrack, where raw dst_key contains many
    essentially-unique tokens (tmp random names, sockets, raw IPs).
    """
    s = str(dst_key)
    scheme = (scheme or "none").strip().lower()
    if scheme in ("", "none"):
        return s

    if scheme == "qut":
        # QUT dst_key space is already fairly small (ports/syscalls/buckets/pattern strings).
        return s

    if scheme != "osp":
        # Unknown scheme: be conservative.
        return s

    # --- OSP normalization ---
    if dst_type == "NET":
        lk = s.lower()
        # Bucket raw IPs to reduce uniqueness.
        if "ip::" in lk:
            ip_s = s.split("ip::", 1)[1]
            try:
                ip = ipaddress.ip_address(ip_s)
                if isinstance(ip, ipaddress.IPv4Address):
                    net = ipaddress.ip_network(f"{ip}/24", strict=False)
                    return f"ip4net::{net.network_address}/24"
                # IPv6: bucket to /64
                net6 = ipaddress.ip_network(f"{ip}/64", strict=False)
                return f"ip6net::{net6.network_address}/64"
            except Exception:
                return "ip::<?>"
        return s

    if dst_type == "FILE":
        # Our canonical FILE keys often look like "...::file::<path>".
        path = s.split("::file::", 1)[1] if "::file::" in s else s
        path_norm = path.lstrip("/")

        # Collapse generic tmp random names but keep meaningful suffixes.
        # Examples: /tmp/avleycoy, /tmp/tmp98cm3s9j
        m = re.match(r"^tmp/(?:tmp)?([A-Za-z0-9]{6,})(\\.[A-Za-z0-9._-]{1,10})?$", path_norm)
        if m:
            suffix = m.group(2) or ""
            return f"tmp/<RAND>{suffix}"

        # Collapse pip temp dirs.
        if path_norm.startswith("tmp/pip-"):
            return "tmp/pip-*"

        # Collapse anon sockets/fds.
        if path_norm.startswith("socket:["):
            return "socket:[*]"
        if path_norm.startswith("anon_inode:["):
            return "anon_inode:[*]"
        if path_norm.startswith("pipe:["):
            return "pipe:[*]"

        # Reduce __pycache__ variance to module-level bucket.
        if "__pycache__" in path_norm:
            return "__pycache__/*"

        return path_norm

    if dst_type == "CMD":
        lk = s.lower()
        # Collapse analysis tool invocations that are environment artifacts.
        if "analyze-python.py" in lk:
            return "analyze-python.py"
        if "analyze-node.js" in lk:
            return "analyze-node.js"
        # Collapse rustc --version probe
        if "rustc" in lk and "--version" in lk:
            return "rustc --version"
        # sleep probes
        if "sleep" in lk:
            return "sleep"
        return s

    return s


def load_rarity_stats(path: str | Path) -> RarityStats:
    p = Path(path)
    obj = json.loads(p.read_text())
    n = int(obj.get("num_graphs", 0))
    normalize = str(obj.get("normalize", "none"))
    raw_df = obj.get("df", {})
    df: Dict[Tuple[EType, str], int] = {}
    for k, v in raw_df.items():
        # key format: "<SRC|REL|DST>\t<dst_key>"
        et_s, dst = k.split("\t", 1)
        df[(str_to_etype(et_s), dst)] = int(v)
    return RarityStats(num_graphs=n, df=df, normalize=normalize)


def save_rarity_stats(path: str | Path, stats: RarityStats) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw_df = {f"{etype_to_str(et)}\t{dst}": int(v) for (et, dst), v in stats.df.items()}
    obj = {"num_graphs": int(stats.num_graphs), "normalize": str(stats.normalize), "df": raw_df}
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

