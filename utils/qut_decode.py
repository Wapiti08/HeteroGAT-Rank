from __future__ import annotations

from typing import Any, List

import ast


def parse_listish(x: Any) -> List[str]:
    """Parse QUT list-like fields into list[str].

    QUT processed CSVs sometimes store lists as:
    - "a, b, c"
    - "['a', 'b']"
    - NaN / empty
    """
    if x is None:
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []

    # Try Python literal first.
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(i).strip() for i in v if str(i).strip()]
        except Exception:
            pass

    # Fallback: comma-separated
    if "," in s:
        return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
    return [s.strip().strip('"').strip("'")]

