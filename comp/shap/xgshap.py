import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

data_path = Path.cwd().parent.joinpath("corr_ana", "corr_matrix.csv")

features_df = pd.read_csv(data_path)

# ---- STEP 1: Prepare data ----
# Assume `features_df` already exists and contains 'Label'
X = features_df.drop(columns=['Label'])
y = features_df['Label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- STEP 2: Train XGBoost Model ----
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# ---- STEP 3: SHAP Explainer ----
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# ---- STEP 4: Plot Summary ----
# Summary bar plot (mean absolute SHAP values per feature)
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Optional: summary dot plot (shows distribution and direction)
shap.summary_plot(shap_values, X_test)
