import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from boruta import BorutaPy
import warnings
import os
from imblearn.under_sampling import RandomUnderSampler
import datetime
warnings.filterwarnings('ignore')

class ModelEvaluation:
    def __init__(self, filepath, target_cols, score_cols, model_params, configs):
        self.filepath = filepath
        self.target_cols = target_cols
        self.score_cols = score_cols
        self.model_params = model_params
        self.configs = configs
        self.metrics = pd.DataFrame(columns=['Model', 'Target', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'Correction', 'Dimension Reduction', 'Cost-sensitive'])
        self.pca_transformers = {}
        self.boruta_features = {}
        self.version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def load_data(self):
        df = pd.read_csv(self.filepath)
        df = df.drop(columns=self.score_cols).reset_index(drop=True)
        return df

    def preprocess(self, df, target_col):
        X = df.drop(columns=self.target_cols)
        if target_col == 'MRS90':
            y = df[target_col].apply(lambda x: 0 if x <= 3 else 1)
        else:
            y = df[target_col]
        return X, y

    def correct_imbalance(self, X, y, method='SMOTE'):
        if method == 'SMOTE':
            resampler = SMOTE(random_state=42)
        elif method == 'undersampling':
            resampler = RandomUnderSampler(random_state=42)
        X_res, y_res = resampler.fit_resample(X, y)
        return X_res, y_res

    def reduce_dimension(self, X, y=None, config=None, target_col=None):
        if not config['dimension_reduction']:
            return X

        if config['dimension_reduction_technique'] == 'PCA':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if target_col not in self.pca_transformers:
                pca = PCA(n_components=0.95)
                self.pca_transformers[target_col] = pca.fit(X_scaled)
            X_reduced = self.pca_transformers[target_col].transform(X_scaled)
        elif config['dimension_reduction_technique'] == 'Boruta' and y is not None:
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
            boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)
            boruta_selector.fit(X.values, y.values)
            self.boruta_features[target_col] = X.columns[boruta_selector.support_].tolist()
            X_reduced = X.loc[:, self.boruta_features[target_col]]
        else:
            X_reduced = X.loc[:, self.boruta_features[target_col]] if config['dimension_reduction_technique'] == 'Boruta' else X
        return X_reduced

    def initialize_models(self, use_cost_sensitive):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced' if use_cost_sensitive else None),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'Random Forest': RandomForestClassifier(class_weight='balanced' if use_cost_sensitive else None),
            # 'Support Vector Machine': CalibratedClassifierCV(SVC(kernel='linear', probability=True, tol=0.01, class_weight='balanced' if use_cost_sensitive else None))
        }
        return models

    def evaluate_model(self, X, y, model_name, target_name, config):
        resampling_method = 'None'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if config['imbalance_correction']:
            resampling_method = 'SMOTE' if 'resampling_strategy' not in config else config['resampling_strategy']
            X_train, y_train = self.correct_imbalance(X_train, y_train, method=resampling_method)
        if config['dimension_reduction']:
            X_train = self.reduce_dimension(X_train, y_train if config['dimension_reduction_technique'] == 'Boruta' else None, config, target_col=target_name)
            X_test = self.reduce_dimension(X_test, config=config, target_col=target_name)

        model = self.initialize_models(config['cost_sensitive'])[model_name]
        grid_search = GridSearchCV(model, self.model_params[model_name], cv=10, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        # Save feature importance if the model is XGBoost or Random Forest
        if model_name in ['XGBoost', 'Random Forest']:
            best_model = grid_search.best_estimator_
            self.save_feature_importance(best_model, X_train.columns.tolist(), model_name, target_name, config)

        if config['dimension_reduction_technique'] == 'Boruta':
            # It's important to ensure the Random Forest used here is the same as used in Boruta
            rf = grid_search.best_estimator_ if isinstance(grid_search.best_estimator_, RandomForestClassifier) else RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
            boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)
            boruta_selector.fit(X.values, y.values)
            self.plot_boruta_results(rf, X.columns.tolist(), boruta_selector.support_, model_name, target_name, config)

        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        self.metrics = self.metrics.append({
            'Model': model_name,
            'Target': target_name,
            'AUC Score': auc_score,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Correction': resampling_method,
            'Dimension Reduction': config['dimension_reduction_technique'],
            'Cost-sensitive': 'Yes' if config['cost_sensitive'] else 'No'
        }, ignore_index=True)

        print(f"Results for {model_name} on target {target_name} with config: {config}")
        print(self.metrics.iloc[-1])
        print()

    def run_evaluation(self):
        df = self.load_data()
        for config in self.configs:
            for target_col in self.target_cols:
                try:
                    X, y = self.preprocess(df, target_col)
                    for model_name in self.initialize_models(config['cost_sensitive']).keys():
                        self.evaluate_model(X, y, model_name, target_col, config)
                except Exception as e:
                    print(f"Error encountered for target {target_col} with config {config}. Error: {e}")
                    continue
    def display_metrics(self, filename=f'metrics.csv'):
        print(self.metrics)
        self.metrics.to_csv(filename, index=False)

    def save_feature_importance(self, model, features, model_name, target_name, config, plot_type="feature_importance"):
        folder = 'plots'
        if not os.path.exists(folder):
            os.makedirs(folder)
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        indices = np.argsort(importances)[::-1]
        
        # Construct a unique filename for each plot
        filename = f"{folder}/{target_name}_{model_name}_{config['dimension_reduction_technique']}_{config['resampling_strategy']}_{plot_type}.png"
        
        plt.figure(figsize=(10, 6))
        plt.title(f'{model_name} Feature Importances for {target_name}')
        plt.bar(range(len(indices)), importances[indices], color='b', align='center')
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_boruta_results(self, model, features, support, model_name, target_name, config):
        folder = 'plots'
        if not os.path.exists(folder):
            os.makedirs(folder)
        importances = model.feature_importances_
        indices = np.argsort(importances)
        colors = ['green' if support[i] else 'red' for i in indices]

        # Construct a unique filename for each plot
        filename = f"{folder}/{target_name}_{model_name}_{config['dimension_reduction_technique']}_{config['resampling_strategy']}_boruta.png"
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Boruta Feature Selection Results for {model_name} on {target_name}')
        plt.bar(range(len(indices)), importances[indices], color=colors)
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    



model_params = {
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    'XGBoost': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
    # 'Support Vector Machine': {'base_estimator__C': [0.1, 1, 10], 'base_estimator__kernel': ['linear']}
}

# Define configurations for running evaluations, including new 'resampling_strategy' key
configs = [
    # Basic configurations without imbalance correction or dimension reduction
    {'imbalance_correction': False, 'dimension_reduction': False, 'cost_sensitive': False, 'dimension_reduction_technique': 'None', 'resampling_strategy': 'None'},

    {'imbalance_correction': True, 'dimension_reduction': False, 'cost_sensitive': False, 'dimension_reduction_technique': 'None', 'resampling_strategy': 'SMOTE'},
    {'imbalance_correction': True, 'dimension_reduction': False, 'cost_sensitive': False, 'dimension_reduction_technique': 'None', 'resampling_strategy': 'undersampling'},

    {'imbalance_correction': False, 'dimension_reduction': True, 'cost_sensitive': False, 'dimension_reduction_technique': 'Boruta', 'resampling_strategy': 'None'},
    {'imbalance_correction': False, 'dimension_reduction': True, 'cost_sensitive': False, 'dimension_reduction_technique': 'PCA', 'resampling_strategy': 'None'},
]

evaluation = ModelEvaluation(filepath='../ich_data_w_scores_modified.csv', 
                             target_cols=['MORT90', 'MRS90'], 
                             score_cols=["oICH_score", "mICH_score", "ICH_GS_score", "LSICH_score", "ICH_FOS_score", "Max_ICH_score"], 
                             model_params=model_params,
                             configs=configs)
evaluation.run_evaluation()
version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
evaluation.display_metrics(filename=f"metrics_{version}.csv")

