from fea import extract_graph_features
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def correlation_analysis(features_df, label_column='Label', method='pearson', visualize=True):
    '''
    Compute and visualize the correlation matrix from a feature DataFrame.

    Parameters:
    - features_df: pandas DataFrame with numerical features and optional label column
    - label_column: name of the label column (must be in features_df if included)
    - method: correlation method ('pearson', 'spearman', etc.)
    - visualize: if True, plot the heatmap

    Returns:
    - corr_matrix: pandas DataFrame of correlation coefficients
    '''
    # Sanity check
    if label_column not in features_df.columns:
        raise ValueError(f"'{label_column}' not found in features_df columns.")

    # Compute correlation
    corr_matrix = features_df.corr(method=method)

    # Plot heatmap
    if visualize:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix ({method})')
        plt.tight_layout()
        plt.show()

    # Save correlation matrix
    corr_path = Path.cwd().joinpath("corr_matrix.csv")
    corr_matrix.to_csv(corr_path)

    return corr_matrix

if __name__ == "__main__":
    data_path = Path.cwd().joinpath("feature_matrix.csv")
    fea_df = pd.read_csv(data_path)
    corr_matrix = correlation_analysis(fea_df, label_column='Label', method='pearson', visualize=True)
    print(corr_matrix)
