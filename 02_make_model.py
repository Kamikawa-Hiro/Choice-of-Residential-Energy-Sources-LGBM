import pickle
import functools
import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RESULT_DIR = Path("result")

# Set seed for reproducibility
SEED = 1

def prepare_xy_splits(df, target_col="house_type"):
    
    # 1. Initialize and fit LabelEncoder
    le = LabelEncoder()
    le.fit(df[target_col])
    
    # 2. Display the mapping for transparency in research
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"LabelEncoder Mapping for '{target_col}':")
    print(f"  {mapping}")
    
    # 3. Split into features and target
    X = df.drop(columns=[target_col])
    y = le.transform(df[target_col])
    
    print(f"\nPreprocessing complete:")
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {X.shape[1]} columns")
    
    return X, y

# --- Execution ---
# Load the data (ensure mock_data.csv is in your directory)
df_mock = pd.read_csv(RESULT_DIR/'mock_data.csv')

# Run the function
X, y = prepare_xy_splits(df_mock, target_col="house_type")

# Optunaによるハイパーパラメータチューニングの関数
def objective_logistic(trial, X_data, y_data,seed):
    # ロジスティック回帰のハイパーパラメータ空間を定義
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    
    # solverに依存するパラメータの設定
    if solver == 'liblinear':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    else:
        penalty = 'l2'  # lbfgsはl2（またはNone）のみ

    params = {
        'C': trial.suggest_float('C', 1e-4, 100.0, log=True), # 範囲を少し広げると安定します
        'solver': solver,
        'penalty': penalty,
        'random_state': seed,
        'max_iter': 2000, # 小規模データでも収束を保証するために少し多めに設定
    }
    
    # 交差検定を行う
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    scores = []
    for train_index, val_index in rskf.split(X_data, y_data):

        X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) 

        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        
        y_pred_proba = model.predict_proba(X_val_scaled) 
        score = roc_auc_score(y_val, y_pred_proba[:, 1])
        scores.append(score)
        
    return np.mean(scores)

# Optunaによるハイパーパラメータチューニングの関数
def objective_lgbm(trial, X_data, y_data,seed):
    # LightGBMのハイパーパラメータ空間を定義
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        # 'is_unbalance': True,
        'random_state': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,    
        'bagging_freq': 1,
        'n_estimators': trial.suggest_int('n_estimators', 10000,15000), # 変更なし (Early Stoppingで制御)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True), # 変更なし
        # --- 1. 木の複雑さを直接制限する ---
        'num_leaves': trial.suggest_int('num_leaves', 4, 8), # 葉の数の上限を大幅に減らす (例: 127 -> 50)
        'max_depth': trial.suggest_int('max_depth', 3, 4), # 木の深さの上限を大幅に減らす (例: 20 -> 10)
        # --- 2. 分岐・葉の生成を厳しくする ---
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 30), # 葉に必要な最小サンプル数を増やす (例: 3 -> 15)
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.03, 1.0, log=True),
        # --- 3. 正則化を強める ---
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True), # L1正則化の最小値を上げ、探索範囲を広げる
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True), # L2正則化の最小値を上げ、探索範囲を広げる
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 0.9, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0),
    }
    # 交差検定を行う
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    scores = []
    
    for train_index, val_index in rskf.split(X_data, y_data):

        X_train, X_val = X_data.iloc[train_index], X_data.iloc[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]
        
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50,verbose=False),
                           lgb.log_evaluation(0)]) # 早期停止（10回改善がなければ停止）

        y_pred_proba = model.predict_proba(X_val) 

        # 2D 配列の「1列目」（クラス1の確率）を指定する
        score = roc_auc_score(y_val, y_pred_proba[:, 1])
        scores.append(score)

    return np.mean(scores)


def run_nested_cv_training(X, y, result_dir, seed):
    
    # --- 0. Initialization ---
    logi_models_list = []
    lgbm_models_list = []
    X_train_scaled_list = []
    X_train_list = []
    X_test_scaled_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # Base parameters that are not subject to tuning
    logi_base_params = {
        'random_state': seed,
        'max_iter': 2000,
    }

    lgbm_base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,        
        'bagging_freq': 1,
        'n_jobs': -1,
    }

    # --- 1. Outer Loop Setup (Evaluation CV) ---
    N_OUTER_SPLITS = 3
    N_OPTUNA_TRIALS = 200
    skf_outer = StratifiedKFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)

    print(f"--- Starting {N_OUTER_SPLITS}-Fold Nested Cross-Validation ---")

    for fold, (train_index, test_index) in enumerate(skf_outer.split(X, y)):
        print(f"\n===== Outer Fold {fold + 1} / {N_OUTER_SPLITS} =====")
        
        # 1a. Split Outer Data
        X_train_outer, X_test_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_test_outer = y[train_index], y[test_index]

        # --- 2. Inner Loop: Hyperparameter Tuning via Optuna ---
        print(f"--- 2. Starting Optuna Tuning (Trials: {N_OPTUNA_TRIALS}) ---")
        
        # 2a. Logistic Regression Tuning
        obj_logi = functools.partial(
            objective_logistic, # Ensure this function is defined in your script
            X_data=X_train_outer, 
            y_data=y_train_outer, 
            seed=seed
        )
        study_logi = optuna.create_study(direction='maximize')
        study_logi.optimize(obj_logi, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
        
        best_logi_params = study_logi.best_params.copy()
        best_logi_params.update(logi_base_params)
        
        # 2b. LightGBM Tuning
        obj_lgbm = functools.partial(
            objective_lgbm, # Ensure this function is defined in your script
            X_data=X_train_outer, 
            y_data=y_train_outer, 
            seed=seed
        )
        study_lgbm = optuna.create_study(direction='maximize')
        study_lgbm.optimize(obj_lgbm, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

        best_lgbm_params = study_lgbm.best_params.copy()
        best_lgbm_params.update(lgbm_base_params)
        
        # --- 3. Model Retraining with Best Parameters ---
        print("--- 3. Retraining models on Outer Train data ---")
        
        # Split training data further for validation (used for early stopping in LightGBM)
        X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
            X_train_outer, y_train_outer,
            test_size=0.25, 
            random_state=seed,
            stratify=y_train_outer
        )
        
        # Feature Scaling (Crucial for Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fit)
        X_test_scaled = scaler.transform(X_test_outer) 
        
        # Train Logistic Regression
        logi_model = LogisticRegression(**best_logi_params)
        logi_model.fit(X_train_scaled, y_train_fit)
        print(f"Logistic: Best Params for Fold {fold+1}: {study_logi.best_params}")

        # Train LightGBM with Early Stopping
        lgbm_model = lgb.LGBMClassifier(**best_lgbm_params)
        lgbm_model.fit(
            X_train_fit, 
            y_train_fit,
            eval_set=[(X_val_fit, y_val_fit)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        print(f"LightGBM: Best Params for Fold {fold+1}: {study_lgbm.best_params}")
        
        # --- 4. Store Results ---
        logi_models_list.append(logi_model)
        lgbm_models_list.append(lgbm_model)
        X_train_list.append(X_train_fit)
        X_train_scaled_list.append(X_train_scaled)
        X_test_list.append(X_test_outer)
        X_test_scaled_list.append(X_test_scaled)
        y_train_list.append(y_train_fit)
        y_test_list.append(y_test_outer)

    # --- 5. Export Results ---
    nested_cv_results = {
        'logi_models': logi_models_list,
        'lgbm_models': lgbm_models_list,
        'X_trains_scaled': X_train_scaled_list,
        'X_trains': X_train_list,
        'X_tests_scaled': X_test_scaled_list,
        'X_tests': X_test_list,
        'y_trains': y_train_list,
        'y_tests': y_test_list
    }

    save_path = result_dir / "trained_models_nested_cv.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(nested_cv_results, f)
    
    print(f"\nSuccess! Models and test data saved to: {save_path}")

# Run the training
run_nested_cv_training(X, y, RESULT_DIR, seed=SEED)