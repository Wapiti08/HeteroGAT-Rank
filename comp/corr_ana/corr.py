from fea import extract_graph_features
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def correlation_analysis(df, label_column='Label', method='pearson', visualize=True):
    # Step 1: Extract features for each row
    features_df = df.apply(extract_graph_features, axis=1)

    # Step 2: Optionally include the label
    if label_column in df.columns:
        features_df['Label'] = df[label_column]

    # Step 3: Compute correlation
    corr_matrix = features_df.corr(method=method)

    # Step 4: Plot heatmap
    if visualize:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix ({method})')
        plt.tight_layout()
        plt.show()

    return corr_matrix

if __name__ == "__main__":
    data_path = Path.cwd().parent.parent.joinpath("data", "label_data.pkl")
    df = pd.read_pickle(data_path)
    corr_matrix = correlation_analysis(df, label_column='Label', method='pearson', visualize=True)
    print(corr_matrix)
