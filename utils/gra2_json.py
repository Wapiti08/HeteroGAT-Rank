'''
 # @ Create Time: 2025-06-02 10:19:11
 # @ Modified time: 2025-06-02 10:19:13
 # @ Description: helper function to convert subgraph inside subgraphs.pkl to JSON format
 '''

import pickle
import json
from pathlib import Path

def gra_to_json(subgraphs, output_dir):
    ''' interface to convert subgraph pickle to json format
    
    '''
    for i, subgraph in enumerate(subgraphs):
        graph_dict = {
            "Nodes": {},
            "Edges": {}
        }

        # construct node_id: f"{eco}::{value}""
        ori_to_id = {}

        for ori_node in subgraph["nodes"]:
            value = ori_node.get("value", "")
            eco = ori_node.get("eco", "")
            if value == "":
                continue

            # create unique node Id
            node_id = f"{eco}::{value}"
            ori_to_id[value] = node_id

            graph_dict["Nodes"][node_id] = {
                "Value": value,
                "Type": ori_node.get("type", ""),
                "Eco": eco
            }

        for edge in subgraph["edges"]:
            src_val = edge["source"]
            tgt_val = edge['target']

            src = ori_to_id.get(src_val)
            tgt = ori_to_id.get(tgt_val)

            if src not in graph_dict["Nodes"] or tgt not in graph_dict["Nodes"]:
                continue

            if src not in graph_dict["Edges"]:
                graph_dict["Edges"][src] = {}

            graph_dict["Edges"][src][tgt] = {
                "Source": src,
                "Target": tgt,
                "Value": edge.get("value", None),
                "Type": edge.get("type", "")
            }

        graph_dict["Label"] = subgraph.get("label", 0)  
        # save to JSON file
        json_path = Path(output_dir).joinpath(f"subgraph_{i}.json")
        with open(json_path, "w") as f_out:
            json.dump(graph_dict, f_out, indent=2)
    
    print(f"✅ {len(subgraphs)} subgraphs converted with unique node IDs (eco::value) in: {output_dir}")


if __name__ == "__main__":
    data_path = Path.cwd().parent.joinpath("data", "subgraphs.pkl")
    with data_path.open("rb") as fr:
        subgraphs = pickle.load(fr)[:50]      

    output_dir = Path.cwd().parent.joinpath("comp", "go_entropy_ana", "sample")
    gra_to_json(subgraphs, output_dir)


