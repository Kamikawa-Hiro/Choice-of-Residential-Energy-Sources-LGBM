import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_auc_score, balanced_accuracy_score, f1_score

RESULT_DIR = Path("result") 


def load_evaluation_data(file_path):
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    print(f"Loading results from {file_path}...")
    
    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    # Extract lists from the dictionary
    logi_models = results['logi_models']
    lgbm_models = results['lgbm_models']
    X_trains_scaled = results['X_trains_scaled']
    X_trains = results['X_trains']
    X_tests_scaled = results['X_tests_scaled']
    X_tests = results['X_tests']
    y_trains = results['y_trains']
    y_tests = results['y_tests']

    print("Successfully loaded:")
    print(f"  - Number of Folds: {len(logi_models)}")
    print(f"  - Models: Logistic Regression, LightGBM")
    
    return logi_models, lgbm_models,X_trains_scaled, X_trains, X_tests_scaled, X_tests, y_trains, y_tests

# --- Execution Example ---
# Define your result directory and file name
save_path = RESULT_DIR / "trained_models_nested_cv.pkl"
#Load all data
logi_models, lgbm_models,X_trains_scaled, X_trains, X_tests_scaled, X_tests, y_trains, y_tests = load_evaluation_data(save_path)

def evaluate_saved_models(logi_models, lgbm_models, 
                          X_trains_scaled, X_trains, 
                          X_tests_scaled, X_tests, 
                          y_trains, y_tests):
    cv_results = []
    n_folds = len(logi_models)

    print(f"--- Starting Evaluation for {n_folds} Folds ---")

    for fold in range(n_folds):
        # 1. Extract data and models for the current fold
        # Logistic Regression Data
        X_tr_scaled = X_trains_scaled[fold]
        X_te_scaled = X_tests_scaled[fold]
        
        # LightGBM Data
        X_tr_raw = X_trains[fold]
        X_te_raw = X_tests[fold]
        
        # Ground Truth
        y_tr = y_trains[fold]
        y_te = y_tests[fold]

        # Models
        m_logi = logi_models[fold]
        m_lgbm = lgbm_models[fold]

        # 2. Define Helper Function for Metrics
        def calculate_scores(model, X_data, y_true):
            y_pred = model.predict(X_data)
            y_prob = model.predict_proba(X_data)[:, 1]
            return {
                'AUC': roc_auc_score(y_true, y_prob),
                'B_ACC': balanced_accuracy_score(y_true, y_pred),
                'F1': f1_score(y_true, y_pred, average='macro')
            }

        # 3. Compute Metrics for each model and split
        # Logistic Regression scores
        scores_logi_train = calculate_scores(m_logi, X_tr_scaled, y_tr)
        scores_logi_test = calculate_scores(m_logi, X_te_scaled, y_te)

        # LightGBM scores
        scores_lgbm_train = calculate_scores(m_lgbm, X_tr_raw, y_tr)
        scores_lgbm_test = calculate_scores(m_lgbm, X_te_raw, y_te)

        # 4. Store into result list
        model_eval_data = [
            ('Logistic', scores_logi_train, scores_logi_test),
            ('LightGBM', scores_lgbm_train, scores_lgbm_test)
        ]

        for model_name, train_s, test_s in model_eval_data:
            for metric in ['AUC', 'B_ACC', 'F1']:
                cv_results.append({
                    'Model': model_name, 'Fold': fold + 1, 'Metric': metric, 
                    'Split': 'Train', 'Score': train_s[metric]
                })
                cv_results.append({
                    'Model': model_name, 'Fold': fold + 1, 'Metric': metric, 
                    'Split': 'Test', 'Score': test_s[metric]
                })

    # --- 5. Data Formatting and Summary ---
    raw_df = pd.DataFrame(cv_results)
    
    # Create pivot table for a clean comparison
    results_df = raw_df.pivot_table(
        index=['Model', 'Fold'], 
        columns=['Metric', 'Split'], 
        values='Score'
    )

    # Reorder columns for a professional look
    metric_order = ['AUC', 'B_ACC', 'F1']
    split_order = ['Train', 'Test']
    results_df = results_df.reindex(columns=metric_order, level=0).reindex(columns=split_order, level=1)

    print("\n===== Performance Summary per Fold =====")
    display(results_df)

    summary_mean = results_df.groupby(level='Model').mean()
    print("\n--- Average Scores Across All Folds ---")
    display(summary_mean)

    return results_df, summary_mean

# --- Execution ---
results_df, summary_mean = evaluate_saved_models(
    logi_models, lgbm_models, X_trains_scaled, X_trains, 
    X_tests_scaled, X_tests, y_trains, y_tests
)

def plot_confusion_matrix(y_true, y_pred, model_name, fold_idx):
    """
    Plots a confusion matrix heatmap for a specific model and fold.
    """
    # Define class labels (ensure these match your LabelEncoder mapping)
    # Mapping: 0 -> All electric, 1 -> Dual fuel
    labels = ['All electric', 'Dual fuel']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization settings
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        annot_kws={'size': 16},
        xticklabels=labels,
        yticklabels=labels
    )
    
    # Formatting titles and labels for publication quality
    plt.title(f'{model_name}: Fold {fold_idx} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout() # Adjust layout to prevent clipping
    plt.show()

# --- Execution: Evaluation per Fold ---
n_folds = len(y_tests)

for fold in range(n_folds):
    print(f"\n--- Evaluating Fold {fold + 1} ---")
    y_true = y_tests[fold]
    
    # 1. Logistic Regression Evaluation
    model_logi = logi_models[fold]
    X_test_logi = X_tests_scaled[fold]
    y_pred_logi = model_logi.predict(X_test_logi)
    
    # Corrected arguments: (y_true, y_pred, model_name, fold_idx)
    plot_confusion_matrix(y_true, y_pred_logi, 'Logistic Regression', fold + 1)

    # 2. LightGBM Evaluation
    model_lgbm = lgbm_models[fold]
    X_test_lgbm = X_tests[fold]
    y_pred_lgbm = model_lgbm.predict(X_test_lgbm)
    
    # Corrected arguments: (y_true, y_pred, model_name, fold_idx)
    plot_confusion_matrix(y_true, y_pred_lgbm, 'LightGBM', fold + 1)