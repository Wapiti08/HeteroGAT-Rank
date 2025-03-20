'''
 # @ Create Time: 2025-03-12 11:41:27
 # @ Modified time: 2025-03-12 11:41:43
 # @ Description: 
 
    - extract subgraphs for different ecosystems
    - compare feature importance using the learned masks
    - rank important nodes/edges for intra-vs. inter-ecosystem connections

 '''

import sys
from pathlib import Path
sys.path.insert(0, Path(sys.path[0]).parent.as_posix())
import torch
from model import mgan
from sklearn.preprocessing import MinMaxScaler


class EcosystemFeatureAnalysis:

    def __init__(self, model:mgan.MaskedHeteroGAT, eco_dict:dict, threshold:float):
        '''
        args:
            model: trained maskedheteroGAT model
            eco_dict: dict mappping package_name nodes to their ecosystems {node_id: ecosystem_label}
        '''
        self.model = model
        self.eco_dict = eco_dict
        self.threshold = threshold
        self.scaler = MinMaxScaler()

    def extract_subgraphs(self, edge_index_dict, node_type):
        ''' extract intra- and inter-ecosystem subgraphs
        
        '''
        intra_edges = {}
        inter_edges = []

        for edge_type, edge_index in edge_index_dict.items():
            # match the source type
            if edge_type[0] == node_type:
                for i in range(edge_index.shape[1]):
                    # extract src and dst
                    src, dst = edge_index[:,i].tolist()
                    # identify the ecosystem
                    eco = self.eco_dict.get[src]
                    # match the ecosystem
                    if eco is not None and eco == self.eco_dict.get(dst):
                        # ensure the ecosystem entry exists
                        if eco not in intra_edges:
                            intra_edges[eco] = []
                        intra_edges[eco].append((src, dst))
                    else:
                        inter_edges.append((src, dst))
        
        return intra_edges, inter_edges

    
    def extract_attens(self, ):
        ''' extract feature importance for nodes and edges
        
        '''    
        node_att = self.model.ext_node_att()
        edge_att = self.model.ext_edge_att()

        return node_att, edge_att


    def diff_ana_eco(self, edge_index_dict, node_type):
        ''' identify critical nodes/edges for intra- and inter-ecosystem malicious labels
        
        '''
        # get the intra- and inter- edges
        intra_edges, inter_edges = self.extract_subgraphs(edge_index_dict, node_type)

        # get the node and edge attentions
        node_att, edge_att = self.extract_attens()
        
        # identify critical nodes --- highly correlated with malicious labels
        intra_nodes = {edge: edge_att[edge] for eco, edges in intra_edges.items() for edge in edges if edge in edge_att}
        inter_nodes = {edge: edge_att[edge] for edge in inter_edges if edge in edge_att}

        # normalize importance scores
        intra_values, inter_values = list(intra_values.values()), list(inter_values.values())
        all_scores = intra_values + inter_values
        if all_scores:
            normalized = self.scaler.fit_transform([ [s] for s in all_scores]).flatten()
            intra_attens = {edge_att: normalized[i] for i, edge_att in enumerate(intra_attens)}
            inter_attens = {edge_att: normalized[len(intra_attens) + i] for i, edge_att in enumerate(inter_attens)}

        # Filter critical edges (importance above threshold)
        critical_intra_edges = {edge: score for edge, score in intra_attens.items() if score >= self.threshold}
        critical_inter_edges = {edge: score for edge, score in inter_attens.items() if score >= self.threshold}

        # identify critical edges per ecosystem
        critical_intra_edges_per_eco = {eco: {} for eco in intra_edges.keys()}
        for eco, edges in intra_edges.items():
            critical_intra_edges_per_eco[eco] = {edge: critical_intra_edges[edge] for edge in edges if edge in critical_intra_edges}
        
        # identify critical nodes per ecosystem
        critical_intra_nodes_per_eco = {eco: set() for eco in intra_edges.keys()}
        for eco, edges in critical_intra_edges_per_eco.items():
            critical_intra_nodes_per_eco[eco].update(node for edge in edges for node in edge)

        # identify critical inter-nodes
        inter_nodes = {node for edge in critical_inter_edges for node in edge}

        # identify shared critical nodes & edges (common indicators of malicious behavior)
        shared_critical_edges = set(critical_intra_edges.keys()) & set(critical_inter_edges.keys())
        shared_critical_nodes = set().union(*critical_intra_nodes_per_eco.values()) & inter_nodes
        
        return {
            "critical_intra_edges": critical_intra_edges_per_eco,
            "critical_inter_edges": critical_inter_edges,
            "critical_intra_nodes": critical_intra_nodes_per_eco,
            "critical_inter_nodes": inter_nodes,
            "shared_critical_edges": shared_critical_edges,
            "shared_critical_nodes": shared_critical_nodes,
        }

    def rank_edges_and_nodes(self, intra_attens:dict, inter_attens:dict, node_att:dict):
        ''' rank nodes based on learned attention scores
        
        '''
        top_intra_edges = sorted(intra_attens.items(), key=lambda x: x[1], reverse=True)[:10]
        top_inter_edges = sorted(inter_attens.items(), key=lambda x: x[1], reverse=True)[:10]
        top_nodes = sorted(node_att.items(), key=lambda x: x[1], reverse=True)[:10]
        return top_intra_edges, top_inter_edges, top_nodes
    


if __name__ == "__main__":
    