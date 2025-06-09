import os
import torch
from collections import defaultdict
from typing import Tuple
from pathlib import Path

def summarize_graph_structure(fpath: str) -> Tuple[Tuple, str, int]:
    try:
        data = torch.load(fpath, map_location='cpu', weights_only=False)
        label = int(data["label"].item())
        num_nodes = sum([data[node_type].num_nodes for node_type in data.node_types])
        num_edges = sum([data[etype]["edge_index"].shape[1] for etype in data.edge_types])
        file_size_kb = os.path.getsize(fpath) // 1024  # in KB
        return (label, num_nodes, num_edges, file_size_kb), fpath, label
    except Exception as e:
        print(f"⚠️ Error loading {fpath}: {e}")
        return None, None, None

def find_and_remove_duplicates_with_threshold(base_dir: str,
                                              retain_per_group: int = 1,
                                              size_tolerance_kb: int = 4,
                                              label0_min_threshold: int = 7499):
    file_list = [
        os.path.join(base_dir, fname)
        for fname in os.listdir(base_dir)
        if fname.startswith("subgraph") and fname.endswith(".pt")
    ]

    group_map = defaultdict(list)
    label0_files = set()

    for f in file_list:
        key, path, label = summarize_graph_structure(f)
        if key and path:
            label, nodes, edges, size_kb = key
            if label == 0:
                label0_files.add(path)

            matched = False
            for delta in range(-size_tolerance_kb, size_tolerance_kb + 1):
                alt_key = (label, nodes, edges, size_kb + delta)
                if alt_key in group_map:
                    group_map[alt_key].append(path)
                    matched = True
                    break
            if not matched:
                group_map[key].append(path)

    removed = 0
    actually_removed_label0 = 0

    for group_key, files in group_map.items():
        if len(files) > retain_per_group:
            # If group is Label 0, check if enough remaining before deleting
            label = group_key[0]
            to_remove = files[retain_per_group:]

            if label == 0:
                if len(label0_files) - len(to_remove) < label0_min_threshold:
                    print(f"⚠️ Skipping deletion in group {group_key} to preserve label0 >= {label0_min_threshold}")
                    continue

            for f in to_remove:
                os.remove(f)
                removed += 1
                if label == 0:
                    actually_removed_label0 += 1
                    label0_files.discard(f)
                print(f"🗑 Removed {f} from group {group_key}")

    print(f"\n✅ Removed {removed} duplicates (Label 0 removed: {actually_removed_label0}).")
    print(f"✅ Remaining Label 0 samples: {len(label0_files)}")

# Example usage
if __name__ == "__main__":
    data_path = Path.cwd().parent.joinpath("ext", "new", "processed").as_posix()
    find_and_remove_duplicates_with_threshold(
        base_dir=data_path,
        retain_per_group=1,
        size_tolerance_kb=4,
        label0_min_threshold=7314  # <= 你可以根据需要设置成 7500、7600 等
    )