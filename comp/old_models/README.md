## Old models (legacy star-schema pipeline)

This folder contains the **original** HeteroGAT-based models and training scripts
that operate on the **legacy star-shaped graph schema**:

`Package_Name → {Path, DNS Host, Command, IP, Port, Hostnames}`

These are kept for **baseline comparison / reproduction** only.

### Inputs
- Graphs produced by the legacy `ext/` pipeline (e.g., `subgraph_*.pt` created by
  `ext/data_create.py` + `ext/hetergraph.py`).
- Not compatible with the new canonical graphs under `artifacts/*/*.graph.pt`.

### Files
- `hetergat.py`, `dhetergat.py`, `pnhetergat.py`: legacy model definitions
- `mgan.py`, `dgan.py`: legacy training/experiment drivers

