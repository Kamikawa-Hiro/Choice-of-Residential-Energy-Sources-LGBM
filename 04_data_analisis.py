import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

def analyze_shap_importance(models, X_test_list):
    all_shap_values_class1 = []
    all_X_test_features = []
    all_y_test_features = []

    print("\n===== Calculating SHAP values for all folds =====")

    for i, model in enumerate(models):
        X_test = X_test_list[i]
        y_test = y_tests[i]
        # Initialize the TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for the current fold
        # Note: shap_values often returns a list or a multi-dimensional array
        shap_v = explainer.shap_values(X_test)

        # Handle different SHAP output formats to extract Class 1 (Positive) values
        # 1. List format: [class0_values, class1_values]
        if isinstance(shap_v, list):
            v = shap_v[1]
        # 2. 3D Array format: (samples, features, classes)
        elif isinstance(shap_v, np.ndarray) and len(shap_v.shape) == 3:
            v = shap_v[:, :, 1]
        # 3. 2D Array format: Directly returns contribution to log-odds (Class 1)
        else:
            v = shap_v

        all_shap_values_class1.append(v)
        all_X_test_features.append(X_test)
        all_y_test_features.append(y_test)

    # --- Data Aggregation ---
    # Concatenate SHAP values and features across all folds
    aggregated_shap_values = np.concatenate(all_shap_values_class1, axis=0)
    aggregated_X_test = pd.concat(all_X_test_features, axis=0)
    aggregated_y_test = np.concatenate(y_tests, axis=0)

    # --- Plot 1: Summary Plot (Beeswarm/Dot) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        aggregated_shap_values, 
        aggregated_X_test,      
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Summary Plot (All Folds Combined)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Summary Plot (Bar Chart / Importance) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        aggregated_shap_values, 
        aggregated_X_test,      
        show=False,
        plot_type="bar"
    )
    plt.title("Feature Importance based on SHAP Values", fontsize=16)
    plt.tight_layout()
    plt.show()

    return aggregated_shap_values, aggregated_X_test, aggregated_y_test

# --- Execution ---
aggregated_shap, aggregated_X, aggregated_y = analyze_shap_importance(lgbm_models, X_tests)



def analyze_feature_significance(col, aggregated_shap_values, aggregated_X_test, y_true, threshold):
    # --- 1. SHAP Dependence Plot ---
    # Interaction_index=col removes the interaction color bar for a cleaner single-feature look
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        col, 
        aggregated_shap_values, 
        aggregated_X_test, 
        interaction_index=col,
        show=False
    )
    plt.title(f"SHAP Dependence Plot: {col}", fontsize=14)
    plt.show()

    # --- 2. Statistical Preprocessing ---
    # Create a DataFrame for statistical analysis
    df_stat = pd.DataFrame({
        'feature_val': aggregated_X_test[col],
        'is_high': (aggregated_X_test[col] > threshold).astype(int),
        'target': y_true
    })

    # --- 3. Fisher's Exact Test ---
    cross_tab = pd.crosstab(df_stat['is_high'], df_stat['target'])
    odds_ratio_fisher, p_value_fisher = stats.fisher_exact(cross_tab)

    print(f"\n===== Statistical Analysis for {col} (Threshold > {threshold}) =====")
    print("--- 2x2 Contingency Table ---")
    display(cross_tab)
    print(f"\nFisher's Exact Test p-value: {p_value_fisher:.4f}")

    # --- 4. Logistic Regression (for Odds Ratio and Confidence Interval) ---
    # Add constant for intercept
    X_logit = sm.add_constant(df_stat['is_high'])
    logit_model = sm.Logit(df_stat['target'], X_logit).fit(disp=0)
    
    # Extract results
    p_value_logit = logit_model.pvalues['is_high']
    odds_ratio = np.exp(logit_model.params['is_high'])
    conf_int = np.exp(logit_model.conf_int().loc['is_high'])

    print("\n--- Logistic Regression Summary ---")
    print(f"Odds Ratio (OR): {odds_ratio:.3f}")
    print(f"95% CI: [{conf_int[0]:.3f}, {conf_int[1]:.3f}]")
    print(f"Logistic P-value: {p_value_logit:.4f}")

    # --- 5. Simplified Interpretation ---
    significance = "Significant" if p_value_logit < 0.05 else "Not Significant"
    print(f"\nConclusion: The effect of {col} is {significance} (p < 0.05)")

    return {
        'p_value': p_value_logit,
        'odds_ratio': odds_ratio,
        'conf_int': (conf_int[0], conf_int[1])
    }

# --- Execution ---
result_env = analyze_feature_significance(
    "environmental_awareness", 
    aggregated_shap, 
    aggregated_X, 
    aggregated_y,
    threshold=7
)

result_age = analyze_feature_significance(
    "owner_age", 
    aggregated_shap, 
    aggregated_X, 
    aggregated_y,
    threshold=4
)

result_floor = analyze_feature_significance(
    "total_floor_area", 
    aggregated_shap, 
    aggregated_X, 
    aggregated_y,
    threshold=4
)

def analyze_interaction_effect(col, interaction_col, aggregated_shap_values, aggregated_X_test, y_true):
    
    # --- 1. SHAP Interaction Plot ---
    # Visualizes how the effect of 'col' changes depending on the value of 'interaction_col'
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        col, 
        aggregated_shap_values, 
        aggregated_X_test, 
        interaction_index=interaction_col,
        show=False
    )
    plt.title(f"Interaction: {col} x {interaction_col}", fontsize=14)
    plt.show()

    # --- 2. Statistical Data Preparation ---
    # Combine features and target for statistical modeling
    df_analysis = aggregated_X_test.copy()
    df_analysis['target_numeric'] = y_true
    
    # Define subgroup: Binarize household size based on the plot's trend (e.g., <=4 vs 5+)
    df_analysis['size_group'] = df_analysis[col].apply(lambda x: '4- persons' if x <= 4 else '5+ persons')

    # --- 3. Subgroup Analysis (Fisher's Exact Test) ---
    # Test for Main Effect of Household Size
    cross_tab_size = pd.crosstab(df_analysis['size_group'], df_analysis['target_numeric'])
    odds_size, p_size = stats.fisher_exact(cross_tab_size)

    print("\n===== Statistical Test: Main Effect of Household Size =====")
    print(cross_tab_size)
    print(f"Fisher's Exact Test p-value: {p_size:.4f}")
    print(f"Odds Ratio: {odds_size:.2f}")

    # Test for Interaction within a specific subgroup (e.g., 4-person households)
    df_sub = df_analysis[df_analysis[col] == 4].copy()
    if not df_sub.empty:
        # Binarize children count (e.g., 0-1 vs 2+)
        df_sub['child_group'] = (df_sub[interaction_col] >= 2).astype(int)
        cross_tab_child = pd.crosstab(df_sub['child_group'], df_sub['target_numeric'])
        odds_child, p_child = stats.fisher_exact(cross_tab_child)

        print(f"\n===== Subgroup Analysis: Effect of Children within 4-person Households =====")
        print(cross_tab_child)
        print(f"Fisher's Exact Test p-value: {p_child:.4f}")

    # --- 4. Interaction Modeling (Logistic Regression) ---
    # Formal test using an interaction term: target ~ A + B + A*B
    # This proves whether the effect of B depends on the level of A
    formula = f'target_numeric ~ {col} * {interaction_col}'
    model_inter = smf.logit(formula, data=df_analysis).fit(disp=0)

    print("\n===== Logistic Regression with Interaction Term =====")
    print(model_inter.summary())
    
    # Extract p-value of the interaction term
    p_interaction = model_inter.pvalues[f'{col}:{interaction_col}']
    print(f"\nP-value for Interaction Term ({col}:{interaction_col}): {p_interaction:.4f}")
    
    return model_inter

# --- Execution ---
model_interaction = analyze_interaction_effect(
    "living_together", 
    "number_of_children", 
    aggregated_shap, 
    aggregated_X, 
    aggregated_y
)