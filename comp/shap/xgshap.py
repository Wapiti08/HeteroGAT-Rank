import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import psutil
import os
import time
import numpy as np


def train_xgboost_shap(features_df, label_column='Label', save_plot=False, plot_prefix='shap'):
    assert label_column in features_df.columns, "Label column not found."

    # drop non-numeric columns
    drop_cols = [label_column]
    for col in ['name_version', 'Ecosystem']:
        if col in features_df.columns:
            drop_cols.append(col)

    # separate features and labels
    X = features_df.drop(columns=drop_cols)
    y = features_df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # train model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # SHAP analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # visualization 1: feature importance bar
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    if save_plot:
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_bar.png")
    else:
        plt.show()

    # visualization 2: feature importance dot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    if save_plot:
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_dot.png")
    else:
        plt.show()

    # Ecosystem analysis
    if 'Ecosystem' in features_df.columns:
        eco_groups = features_df.groupby('Ecosystem')
        eco_scores = {}

        for eco, group in eco_groups:
            X_eco = group.drop(columns=drop_cols, errors='ignore')
            if len(X_eco) == 0:
                continue
            shap_vals_eco = explainer(X_eco)
            feature_means = np.abs(shap_vals_eco.values).mean(axis=0)
            eco_scores[eco] = dict(zip(X_eco.columns, feature_means))

        # convert to DataFrame
        score_df = pd.DataFrame(eco_scores).T  # shape: ecos x features
        score_df = score_df[[col for col in X.columns if col in score_df.columns]]  # preserve column order

        # compute variance per feature across ecos
        feature_variance = score_df.var(axis=0)
        overall_variance = score_df.values.flatten().var()

        print("\n[Feature Variance Across Ecosystems]")
        print(feature_variance.sort_values(ascending=False))
        print(f"\n[Overall SHAP Score Variance Across All Ecosystems] = {overall_variance:.6f}")

    # compute variance within top-K features per ecosystem
    topk = 10 
    eco_top_variance = {}
    all_top_scores = []
    
    for eco in score_df.index:
        scores = score_df.loc[eco].abs().sort_values(ascending=False)
        top_scores = scores.iloc[:topk]
        eco_top_variance[eco] = top_scores.var()
        all_top_scores.extend(top_scores.tolist())
    
    overall_top_variance = np.var(all_top_scores)
    
    print("\n[SHAP Score Variance Among Top Features Per Ecosystem]")
    for eco, var in eco_top_variance.items():
        print(f"{eco}: {var:.6f}")
    print(f"Overall: {overall_top_variance:.6f}")

    return model, shap_values

if __name__ == "__main__":
    data_path = Path.cwd().parent.joinpath("corr_ana", "feature_matrix.csv")
    assert data_path.exists(), f"File not found: {data_path}"

    fea_df = pd.read_csv(data_path)

    print("[Start] Training with SHAP + XGBoost...")
    start = time.time()

    model, shap_values = train_xgboost_shap(
        fea_df,
        label_column='Label',
        save_plot=True,
        plot_prefix='xgboost_shap'
    )

    end = time.time()
    print(f"[SHAP Analysis] CPU memory usage (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    print(f"[SHAP Analysis] Total time elapsed: {end - start:.2f} seconds")

