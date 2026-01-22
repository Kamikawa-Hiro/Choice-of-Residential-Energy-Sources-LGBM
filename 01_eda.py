import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
RESULT_DIR = Path("result")

##################
# make mock data
##################

# Ensure reproducibility
SEED = 1
np.random.seed(SEED)

NUM_SAMPLES = 240
NUM_ALL_ELECT = 102
NUM_GAS_ELECT = 138
NUM_NA_AGE = 57  # Set 57 out of 240 households (23.8%) to NULL

# 1. Create house_type with a fixed distribution
house_types = (['all_elect'] * NUM_ALL_ELECT) + (['gas_elect'] * NUM_GAS_ELECT)
np.random.shuffle(house_types)

# 2. Define data mapping (owner_age initially excludes NAs for sampling)
data_map = {
    'total_floor_area': ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 'Float64'),
    'owner_last_education': ([1, 2, 3, 4], 'int64'),
    'number_of_cars': ([1, 2, 3, 4], 'Int64'),
    'living_together': ([1, 2, 3, 4, 5, 6, 7, 8], 'int64'),
    'number_of_children': ([0.0, 1.0, 2.0, 3.0, 4.0], 'float64'),
    'owner_age': ([3, 4, 5, 6, 7, 8], 'Int64'),
    'household_income': ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Float64'),
    'environmental_awareness': ([2, 3, 4, 5, 6, 7, 8, 9, 10], 'Int64'),
    'occupation_Civil_Servant': ([0, 1], 'int64'),
    'occupation_Employee': ([0, 1], 'int64'),
    'occupation_Other': ([0, 1], 'int64')
}

# 3. Generate random mock data
mock_dict = {'house_type': house_types}
for col, (values, dtype) in data_map.items():
    mock_dict[col] = np.random.choice(values, size=NUM_SAMPLES)

# Create DataFrame
df_mock = pd.DataFrame(mock_dict)

# Apply data types (using Nullable types like Int64 to handle pd.NA)
for col, (values, dtype) in data_map.items():
    df_mock[col] = df_mock[col].astype(dtype)

# 4. Randomly set owner_age to NULL (pd.NA) for 57 samples
na_indices = np.random.choice(df_mock.index, size=NUM_NA_AGE, replace=False)
df_mock.loc[na_indices, 'owner_age'] = pd.NA

# Verify results
print(f"Total samples: {len(df_mock)}")
print(df_mock['house_type'].value_counts())
print(f"owner_age NULL count: {df_mock['owner_age'].isna().sum()}")
print(df_mock.info())

# Save to CSV
save_name = f"mock_data.csv"
df_mock.to_csv(RESULT_DIR / save_name, index=False)

####################
# Age impute
####################

# --- 1. Configuration ---
target_cols = [
    'total_floor_area', 'owner_last_education', 'number_of_cars', 
    'living_together', 'number_of_children', 'owner_age',
    'household_income', 'environmental_awareness',
    'occupation_Civil_Servant', 'occupation_Employee', 'occupation_Other'
]

# --- 2. Initialize MICE Imputer ---
# Using RandomForestRegressor as the estimator to handle non-linear relationships
mice_imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=SEED
    ),
    max_iter=10,
    random_state=SEED
)

# --- 3. Execute Imputation ---
imputed_array = mice_imputer.fit_transform(df_mock[target_cols])

# --- 4. Post-processing: Clipping and Rounding ---
for i, col in enumerate(target_cols):
    col_min = df_mock[col].min()
    col_max = df_mock[col].max()
    
    # Clip values to ensure they stay within the observed min/max range
    values = np.clip(imputed_array[:, i], col_min, col_max)
    
    # Round to integer and update the array
    imputed_array[:, i] = np.round(values).astype(int)

# --- 5. Update DataFrame ---
df_mock_imputed = df_mock.copy()
df_mock_imputed[target_cols] = imputed_array

# --- 6. Verification ---
print("Imputation completed with clipping and rounding.")
print(f"Missing values in owner_age after imputation: {df_mock_imputed['owner_age'].isna().sum()}")
display(df_mock_imputed[target_cols].head())


# Save to CSV
save_name = f"mock_data_imputed.csv"
df_mock_imputed.to_csv(RESULT_DIR / save_name, index=False)


