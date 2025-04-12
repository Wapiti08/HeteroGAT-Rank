# One-time cleaning script
import torch
import ray
from pathlib import Path

ray.init()

root = Path.cwd().joinpath("test-small", "processed")
pt_files = list(root.glob("batch_*.pt"))

for pt in pt_files:
    data = torch.load(pt)
    if isinstance(data, ray.ObjectRef):
        print(f"[Fix] Converting {pt.name} from Ray ObjectRef")
        resolved = ray.get(data)
        torch.save(resolved, pt)