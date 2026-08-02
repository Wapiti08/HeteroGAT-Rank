"""Write one compact human-review file per runtime dataset."""

from pathlib import Path

import pandas as pd

from .review_queue import REVIEW_COLUMNS


def preserve_review_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if frame.empty or not path.exists():
        return frame
    existing = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "review_id" not in existing:
        return frame
    output = frame.set_index("review_id", drop=False)
    existing = existing.drop_duplicates("review_id", keep="last").set_index("review_id")
    shared = output.index.intersection(existing.index)
    for column in REVIEW_COLUMNS:
        if column not in existing:
            continue
        values = existing.loc[shared, column]
        nonempty = values[values != ""]
        output.loc[nonempty.index, column] = nonempty
    return output.reset_index(drop=True)


def write_review_file(out: Path, frame: pd.DataFrame) -> Path:
    """Replace legacy derived outputs while preserving edits in review.tsv."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "review.tsv"
    frame = preserve_review_columns(frame, path)
    for existing in out.iterdir():
        if existing.is_file() and existing != path:
            existing.unlink()
    frame.to_csv(path, sep="\t", index=False)
    return path
