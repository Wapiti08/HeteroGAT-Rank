from fea import extract_graph_features
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import psutil
import os
import time

def overall_label_correlation(features_df, label_column='Label', method="pearson", visualize=True):
    features_df[label_column] = features_df[label_column].astype(int)
    # get all numeric columns
    numeric_df = features_df.select_dtypes(include=[np.number])

    if label_column not in numeric_df.columns:
        raise ValueError(f"'{label_column}' not found among numeric columns.")

    # compute overall correlation
    corr_matrix = numeric_df.corr(method=method)
    # calculate correlation with label
    label_corr = corr_matrix[label_column].drop(label_column)

    if visualize:
        plt.figure(figsize=(10, 6))
        label_corr.sort_values().plot(kind='barh')
        plt.title(f'Feature Correlation with {label_column} ({method})')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.show()

    label_corr.to_csv("label_feature_corr_overall.csv")
    return label_corr


def per_eco_label_corr(features_df, group_column='Ecosystem', label_column='Label', method='pearson', top_k=10):
    '''
    Calculate per-ecosystem correlation of features with the label.

    '''
    eco_top_features = []
    for eco, group in features_df.groupby(group_column):
        group = group.copy()
        group[label_column] = group[label_column].astype(int)
        numeric_df = group.select_dtypes(include=[np.number])
        if label_column not in numeric_df.columns or numeric_df.shape[0] < 5:
            continue

        corr_matrix = numeric_df.corr(method=method)
        if label_column not in corr_matrix.columns:
            continue

        label_corr = corr_matrix[label_column].drop(label_column)
        top_corr = label_corr.abs().sort_values(ascending=False).head(top_k)

        for feat, corr_val in top_corr.items():
            eco_top_features.append({
                'Ecosystem': eco,
                'Feature': feat,
                'Correlation': label_corr[feat]
            })

    eco_df = pd.DataFrame(eco_top_features)
    eco_df.to_csv("corr_eco_label1.csv")

    return eco_df

if __name__ == "__main__":
    data_path = Path.cwd().joinpath("feature_matrix.csv")
    fea_df = pd.read_csv(data_path)
    start = time.time()

    label_corr = overall_label_correlation(fea_df)
    eco_top_df = per_eco_label_corr(fea_df)

    # [新增] 统计每个 ecosystem 的 top-K correlation 方差
    eco_var_result = {}
    all_scores = []

    for eco, group in eco_top_df.groupby("Ecosystem"):
        scores = group["Correlation"].abs().values
        if len(scores) > 1:
            var = np.var(scores)
            eco_var_result[eco] = var
            all_scores.extend(scores)

    print(f"\n[Correlation Analysis] CPU memory usage (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")

    valid_ecos = ["npm", "pypi", "ruby"]
    filtered_scores = []
    
    print("\n[Top-K Correlation Variance Per Ecosystem]")
    for eco, group in eco_top_df.groupby("Ecosystem"):
        if eco not in valid_ecos:
            continue
        scores = group["Correlation"].abs().values
        if len(scores) > 1:
            var = np.var(scores)
            eco_var_result[eco] = var
            filtered_scores.extend(scores)
            print(f"{eco}: {var:.6f}")
    
    if filtered_scores:
        overall_variance = np.var(filtered_scores)
        print(f"Overall (npm+pypi+ruby): {overall_variance:.6f}")
    else:
        print("Overall (npm+pypi+ruby): N/A (no valid features)")
