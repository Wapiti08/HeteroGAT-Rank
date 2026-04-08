from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class GraphExample:
    path: Path
    y: Optional[int]
    source: str  # e.g. "qut" or "osptrack"
    package: str


def _load_graph_pt(path: Path) -> Tuple[HeteroData, GraphExample]:
    obj = torch.load(path, map_location="cpu")
    data_dict = obj["data_dict"]
    data = HeteroData.from_dict(data_dict)

    y = obj.get("y", None)
    if isinstance(y, torch.Tensor):
        y = int(y.item())
    elif y is not None:
        y = int(y)

    package = str(obj.get("package", path.stem))
    source = "qut" if "qut" in path.as_posix() else "osptrack" if "osp" in path.as_posix() else "unknown"
    ex = GraphExample(path=path, y=y, source=source, package=package)
    return data, ex


class CanonicalGraphDataset(Dataset):
    """Loads canonical graphs saved by scripts into `HeteroData` objects."""

    def __init__(self, graph_dirs: Sequence[str | Path]):
        self.paths: List[Path] = []
        for d in graph_dirs:
            p = Path(d)
            self.paths.extend(sorted(p.glob("*.graph.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No *.graph.pt found under: {graph_dirs}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> HeteroData:
        data, ex = _load_graph_pt(self.paths[idx])
        # store metadata for debugging
        data["meta"] = {"source": ex.source, "package": ex.package}
        # Always attach `y` so PyG collation is consistent across datasets.
        # Use -1 for unknown labels (e.g., some QUT subsets).
        y = ex.y if ex.y is not None else -1
        data["y"] = torch.tensor([y], dtype=torch.long)
        return data

