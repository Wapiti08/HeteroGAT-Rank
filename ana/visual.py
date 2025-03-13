'''
 # @ Create Time: 2025-03-13 12:00:15
 # @ Modified time: 2025-03-13 12:00:37
 # @ Description: visualize critical edges / nodes spanning ecosystems and compare the difference and commons
 '''

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def vis_heatmap(critical_intra_nodes_per_eco:dict):
    eco_names = list(critical_intra_nodes_per_eco.keys())
    node_atten_df = pd.DataFrame(critical_intra_nodes_per_eco).fillna(0)

    # Create a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(node_atten_df, cmap="coolwarm", annot=True)
    plt.title("Comparison of Critical Nodes Across Ecosystems")
    plt.xlabel("Nodes")
    plt.ylabel("Ecosystem")
    plt.show()


def vis_venn(critical_intra_nodes_per_eco:dict):
    ''' understanding overlap of critical nodes/edges between core ecosystems
    
    '''
    set1 = critical_intra_nodes_per_eco["rubygems"]
    set2 = critical_intra_nodes_per_eco["pypi"]
    set3 = critical_intra_nodes_per_eco["npm"]

    plt.figure(figsize=(6, 6))
    venn3([set1, set2, set3], ('NPM', 'PyPI', 'Maven'))
    plt.title("Comparison of Critical Nodes Across Ecosystems")
    plt.show()



def vis_bar(critical_intra_nodes_per_eco:dict):
    ''' rank top critical nodes in each ecosystem
    
    '''
    df = pd.DataFrame([
        {"Ecosystem": eco, "Node": node, "Importance": score}
        for eco, nodes in critical_intra_nodes_per_eco.items()
        for node, score in nodes.items()
    ])

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Node", y="Importance", hue="Ecosystem")
    plt.xticks(rotation=90)
    plt.title("Most Critical Nodes by Ecosystem")
    plt.show()