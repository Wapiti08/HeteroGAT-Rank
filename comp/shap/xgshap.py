import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import psutil
import os

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

    # visualization 1：feature importance bar
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    if save_plot:
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_bar.png")
    else:
        plt.show()

    # visualization 2：feature importance dot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    if save_plot:
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_dot.png")
    else:
        plt.show()

    return model, shap_values


if __name__ == "__main__":
    data_path = Path.cwd().parent.joinpath("corr_ana", "feature_matrix.csv")
    fea_df = pd.read_csv(data_path)
    train_xgboost_shap(fea_df, label_column='Label', save_plot=True, plot_prefix='xgboost_shap')
    print(f"[Correlation Analysis] CPU memory usage (RSS): {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")


