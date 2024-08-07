import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from boruta import BorutaPy
import os
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load

def correct_imbalance(X, y, method='SMOTE'):
    if method == 'SMOTE':
        resampler = SMOTE(random_state=42)
    elif method == 'undersampling':
        resampler = RandomUnderSampler(random_state=42)
    X_res, y_res = resampler.fit_resample(X, y)
    return X_res, y_res

def apply_boruta(X, y, max_depth=5, n_estimators='auto', random_state=42):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=max_depth, random_state=random_state)
    boruta_selector = BorutaPy(rf, n_estimators=n_estimators, random_state=random_state)
    boruta_selector.fit(X.values, y.values)
    selected_features = X.columns[boruta_selector.support_].tolist()
    X_reduced = X.loc[:, selected_features]
    return X_reduced

def apply_pca(X, n_components=0.95):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced

df = pd.read_csv('/Users/umarayub/projects/aizaz_health_project/ich_machine_learning/etc/ich_data_w_scores_modified.csv')

# Drop specified score columns and reset index
score_cols = ["oICH_score", "mICH_score", "ICH_GS_score", "LSICH_score", "ICH_FOS_score", "Max_ICH_score"]
df = df.drop(columns=score_cols).reset_index(drop=True)

# Features and targets
X = df.drop(columns=['MORT90', 'MRS90'])
y_mort90 = df['MORT90']
y_mrs90 = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)  # Binarizing MRS90

# Splitting the dataset into training and temporary test sets
X_train, X_test, y_train_mort90, y_test_mort90 = train_test_split(X, y_mort90, test_size=0.3, random_state=42)
X_train, X_test, y_train_mrs90, y_test_mrs90 = train_test_split(X, y_mrs90, test_size=0.3, random_state=42)

# Apply SMOTE and undersampling after splitting, only to training data
X_train_mort90_smote, y_train_mort90_smote = correct_imbalance(X_train, y_train_mort90, method='SMOTE')
X_train_mrs90_smote, y_train_mrs90_smote = correct_imbalance(X_train, y_train_mrs90, method='SMOTE')
X_train_mort90_undersample, y_train_mort90_undersample = correct_imbalance(X_train, y_train_mort90, method='undersampling')
X_train_mrs90_undersample, y_train_mrs90_undersample = correct_imbalance(X_train, y_train_mrs90, method='undersampling')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Example of applying Boruta
rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
boruta.fit(X_train_scaled, y_train_mort90)
X_train_boruta = boruta.transform(X_train_scaled)
X_test_boruta = boruta.transform(X_test_scaled)

datasets = {
    "Original mort90": (X_train, y_train_mort90, X_test, y_test_mort90),
    "SMOTE mort90": (X_train_mort90_smote, y_train_mort90_smote,  X_test, y_test_mort90),
    "Undersampled mort90": (X_train_mort90_undersample, y_train_mort90_undersample,  X_test, y_test_mort90),
    "Boruta mort90": (X_train_boruta, y_train_mort90,  X_test_boruta, y_test_mort90),
    "PCA mort90": (X_train_pca, y_train_mort90,  X_test_pca, y_test_mort90),
    "Original mrs90": (X_train, y_train_mrs90, X_test, y_test_mrs90),
    "SMOTE mrs90": (X_train_mrs90_smote, y_train_mrs90_smote, X_test, y_test_mrs90),
    "Undersampled mrs90": (X_train_mrs90_undersample, y_train_mrs90_undersample,X_test, y_test_mrs90),
    "Boruta mrs90": (X_train_boruta, y_train_mrs90, X_test_boruta, y_test_mrs90),
    "PCA mrs90": (X_train_pca, y_train_mrs90, X_test_pca, y_test_mrs90)
}

def run_and_evaluate_model(model, param_grid, datasets, model_name, results_df_path):
    # Initialize or load results DataFrame
    try:
        results_df = pd.read_csv(results_df_path)
    except FileNotFoundError:
        results_columns = ['Target', 'Dataset', 'Model', 'Config ID', 'Best Params', 'Test AUC', 'Precision', 'Recall', 'F1']
        results_df = pd.DataFrame(columns=results_columns)

    config_id = 0  # Initialize configuration ID for each model saved

    for name, (X_train, y_train, X_test, y_test) in datasets.items():
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        # Save the best model
        best_model = grid_search.best_estimator_
        y_test_pred = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_test_pred >= 0.5).astype(int)
        auc_test = roc_auc_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save best model to disk
        model_filename = f"models/{model_name}_{name}_best_config_{config_id}.joblib"
        dump(best_model, model_filename)

        # Append best model results
        results_df = results_df.append({
            'Target': 'mrs90' if 'mrs90' in name else 'mort90',
            'Dataset': name,
            'Model': model_name,
            'Config ID': config_id,
            'Best Params': str(grid_search.best_params_),
            'Test AUC': auc_test,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }, ignore_index=True)

        config_id += 1  # Increment config ID for unique filenames

    # Save updated results DataFrame
    results_df.to_csv(results_df_path, index=False)
    return results_df

# Define parameter grids for each model
param_grids = {
    'LogisticRegression': {'C': [0.01, 0.1, 1], 'solver': ['liblinear', 'lbfgs']},
    'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 10]}
}

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier()
}

# Run the evaluation for each model
for model_name, model in models.items():
    results_df_path = f"{model_name.lower()}_results.csv"
    run_and_evaluate_model(model, param_grids[model_name], datasets, model_name, results_df_path)
