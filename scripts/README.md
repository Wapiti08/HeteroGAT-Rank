## Scripts

### Generate canonical QUT graphs (joined by package)

Produces one JSON file per package containing canonical nodes/edges; if the
current Python interpreter has `torch` + `torch_geometric`, it will also emit a
`.graph.pt` file (safe dict via `HeteroData.to_dict()`).

```bash
python scripts/generate_qut_canonical_graphs.py --limit-packages 10
```

Generate a single package:

```bash
python scripts/generate_qut_canonical_graphs.py --package-name "10Cent10-999.0.4.tar.gz"
```

### Generate canonical OSPTrack graphs/events

Streams `data/label_data.csv` in chunks and writes one `.events.json` per row/package
(plus `.graph.pt` if `torch`+PyG are available in the current interpreter).

```bash
python scripts/generate_osptrack_canonical_graphs.py --limit-rows 10
```

