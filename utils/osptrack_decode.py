"""
Helpers for decoding OSPTrack row fields.

In the pickle form (`label_data.pkl`), many columns already contain Python objects
(lists/dicts) and NumPy arrays. This module normalizes them into plain Python
types so downstream parsers can be deterministic and dataset-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import re


def _preprocess_csv_blob(s: str) -> str:
    """Make `label_data.csv` blobs closer to valid Python literals.

    The CSV often contains numpy pretty-printed arrays/lists that look like:
      [{'a': 1}
       {'a': 2}]
    i.e., missing commas between dict items. This function inserts commas where safe.
    """
    # Normalize whitespace/newlines
    s = s.replace("\r\n", "\n")

    # Insert missing commas between adjacent dict literals inside a list.
    # Example: "}\n {" or "} {" -> "}, {"
    s = re.sub(r"\}\s*\{", "}, {", s)
    return s


def _safe_eval_osptrack(s: str) -> Any:
    """Safely evaluate OSPTrack stringified Python-ish structures.

    `label_data.csv` contains values like:
    - "[]"
    - "[{'Path': '/dev/null', 'Read': True, ...}, ...]"
    - "{'Class': 'IN', 'Queries': array([{'Hostname': 'x', 'Types': array(['A'], dtype=object)}], dtype=object)}"

    We do NOT allow builtins; we only provide an `array(...)` shim that returns
    its first positional argument (usually a list).
    """
    s = s.strip()
    if not s:
        return []
    # Quick reject: if it doesn't look like a literal container, don't eval.
    if not (s.startswith("[") or s.startswith("{")):
        return s

    s = _preprocess_csv_blob(s)

    def array(x, dtype=None):  # noqa: ANN001
        return x

    env = {"__builtins__": {}}
    local = {"array": array, "nan": None, "None": None, "True": True, "False": False}
    return eval(s, env, local)  # noqa: S307


def to_py(x: Any) -> Any:
    """Convert numpy scalars/arrays to Python types; leave others unchanged."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def as_list(x: Any) -> List[Any]:
    """Return x as a list; treat None/NaN/empty as [].

    OSPTrack fields might be:
    - [] / list
    - np.ndarray
    - None / NaN
    """
    if x is None:
        return []
    # pandas NaN becomes float('nan')
    try:
        if isinstance(x, float) and np.isnan(x):
            return []
    except Exception:
        pass

    if isinstance(x, list):
        return [to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return [to_py(v) for v in x.tolist()]
    if isinstance(x, str):
        try:
            parsed = _safe_eval_osptrack(x)
        except Exception:
            return []
        return as_list(parsed)
    # Some rows may have a single dict instead of list[dict]
    if isinstance(x, dict):
        return [to_py(x)]
    return []


def as_str_list(x: Any) -> List[str]:
    """Return x as a list[str]."""
    out: List[str] = []
    for v in as_list(x):
        v = to_py(v)
        if v is None:
            continue
        out.append(str(v))
    return out


def get_dict(d: Any, key: str, default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    return to_py(d.get(key, default))


def iter_dicts(x: Any) -> Iterable[Dict[str, Any]]:
    for v in as_list(x):
        if isinstance(v, dict):
            # ensure values are python-native
            yield {k: to_py(val) for k, val in v.items()}


def normalize_command(cmd_value: Any) -> str:
    """OSPTrack command is often list/np.array of argv tokens; normalize to a string."""
    if cmd_value is None:
        return ""
    if isinstance(cmd_value, str):
        return cmd_value.strip()
    if isinstance(cmd_value, (list, tuple, np.ndarray)):
        parts = [str(to_py(p)).strip() for p in list(cmd_value) if str(to_py(p)).strip()]
        return " ".join(parts)
    return str(to_py(cmd_value)).strip()

