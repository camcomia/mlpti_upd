import os
import sys
import argparse
import math
import time
import pandas as pd
import numpy as np
from statistics import mean, median, StatisticsError
from collections import OrderedDict
import glob

# Required ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, BayesianRidge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    # Removed: from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
except ImportError as e:
    print(f"Error: Missing scikit-learn component ({e}).")
    print("Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

# --- Project-Specific Definitions (from feature collection script context) ---
try:
    # Import specific items if needed for file naming
    from defs_PhyAI import SUMMARY_PER_DS, KFOLD as DEF_KFOLD
    # Assuming step=1 for the pattern:
    INPUT_FILENAME_PATTERN = SUMMARY_PER_DS.format("*", "tbr_standard_with_sub_feats", "br", 1)
    # INPUT_FILENAME_PATTERN = "*ds_summary_tbr_standard_with_sub_feats_br_step1.csv" # Simpler pattern
except ImportError:
    print("Warning: Could not import from defs_PhyAI.py. Using fallback pattern for input files.")
    INPUT_FILENAME_PATTERN = "*ds_summary_tbr_standard_with_sub_feats_br_step*.csv" # Fallback pattern
    DEF_KFOLD = 10 # Fallback kfold

# --- Constants based on the Feature Collection Script Output ---
LABEL = "d_ll" # Target variable: likelihood difference
TBR_FEATURE_COLUMNS =[
    'total_bl_orig', 'longest_bl_orig', 'bl_insert_edge1', 'bl_dist_insert', 
    'est_new_bl','total_bl_sub1','topo_dist_insert','bl_insert_edge2','longest_bl_sub1',
    'longest_bl_sub2', 'bl_bisect_edge','leaves_sub1', 'total_bl_sub2', 'leaves_sub2'
]
FEATURES = OrderedDict([(col, col) for col in TBR_FEATURE_COLUMNS])
FEATURES["group_id"] = "orig_ds_id"

FEATURE_COLS = list(TBR_FEATURE_COLUMNS)
KFOLD = DEF_KFOLD
GROUP_ID_COL = FEATURES.get("group_id", "orig_ds_id")
BASELINE_LL_COL = "orig_ds_ll"
# Removed INNER_CV_FOLDS

# Define column types
list_str = ['move_id', 'bisect_branch', 'attach_edge1_node', 'attach_edge2_node', 'reconnect_type', GROUP_ID_COL]
list_int = ['leaves_sub1', 'leaves_sub2', 'topo_dist_insert']
list_float = ['ll', BASELINE_LL_COL, LABEL] + [col for col in FEATURE_COLS if col not in list_int]

types_dict = {}
for e in list_str: types_dict[e] = np.object_
for e in list_int: types_dict[e] = pd.Int64Dtype()
for e in list_float: types_dict[e] = np.float32


# --- Helper Functions ---

def log_experiment(model_name, params, metrics):
    """Logs experiment parameters and metrics (Simplified: no grid search info)."""
    print(f"--- {model_name} (Default Params) Experiment ---")
    if params:
        print(f"  Default Parameters Used: {params}")
    else:
        print("  Default Parameters Used: Sklearn Defaults") # Handle empty default dicts

    printable_metrics = {k: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics.items() if v is not None}
    print(f"  Metrics: {printable_metrics}")
    print("-" * (len(model_name) + 25)) # Adjusted length


# (score_rank, ds_scores, split_features_label, impute_nans_with_train_mean remain unchanged)
def score_rank(df_by_ds, sortby, locatein, random_choice, scale_score):
    """Finds rank of the best predicted item within the true ranking."""
    if df_by_ds.empty or sortby not in df_by_ds or locatein not in df_by_ds: return np.nan
    df_by_ds.loc[:, sortby] = pd.to_numeric(df_by_ds[sortby], errors='coerce')
    df_by_ds.loc[:, locatein] = pd.to_numeric(df_by_ds[locatein], errors='coerce')
    df_by_ds = df_by_ds.dropna(subset=[sortby, locatein])
    if df_by_ds.empty: return np.nan

    if random_choice:
        best_pred_ix = np.random.choice(df_by_ds.index, 1, replace=False)[0]
    else:
        try:
            best_pred_ix = df_by_ds[sortby].idxmax()
        except ValueError:
            return np.nan

    temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()
    rank_list = temp_df.index[temp_df["index"] == best_pred_ix].tolist()
    if not rank_list: return np.nan
    best_pred_rank = min(rank_list) + 1

    if scale_score:
        best_pred_rank = (best_pred_rank / len(temp_df)) * 100 if len(temp_df) > 0 else 0

    return best_pred_rank

def ds_scores(df, target_label, group_id_col, random_choice, scale_score):
    """Calculates ranks and correlations per dataset group."""
    rank_pred_by_ds, rank_test_by_ds, sp_corrs = {}, {}, []
    if group_id_col not in df.columns or 'pred' not in df.columns:
        print(f"Warning: Required columns ('{group_id_col}', 'pred') not found for ds_scores.")
        return rank_pred_by_ds, rank_test_by_ds, sp_corrs

    grouped_df_by_ds = df.groupby(group_id_col, sort=False)
    for group_id, df_by_ds_orig in grouped_df_by_ds:
        df_by_ds = df_by_ds_orig.copy()
        rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", target_label, random_choice, scale_score)
        rank_test_by_ds[group_id] = score_rank(df_by_ds, target_label, "pred", random_choice, scale_score)
        temp_df = df_by_ds[[target_label, "pred"]].dropna()
        if len(temp_df) > 1 and temp_df[target_label].nunique() > 1 and temp_df["pred"].nunique() > 1:
            try:
                corr = temp_df.corr(method='spearman').iloc[0, 1]
                sp_corrs.append(corr)
            except (IndexError, ValueError):
                 sp_corrs.append(np.nan)
        else:
            sp_corrs.append(np.nan)
    return rank_pred_by_ds, rank_test_by_ds, sp_corrs

def split_features_label(df, target_label, feature_cols):
    """Splits DataFrame into features (X) and labels (y)."""
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in DataFrame: {missing_features}")
        return None, None
    if target_label not in df.columns:
        print(f"Error: Target label '{target_label}' not found in DataFrame.")
        return None, None

    numeric_feature_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_feature_cols) != len(feature_cols):
        non_numeric = set(feature_cols) - set(numeric_feature_cols)
        print(f"Warning: Non-numeric columns found in specified features: {non_numeric}. These will be excluded.")

    attributes_df = df[numeric_feature_cols]
    label_df = df[target_label]
    try:
        x = np.array(attributes_df, dtype=np.float32)
    except ValueError as e:
        print(f"Error converting features to numeric array: {e}.")
        # Further debugging: Check which column causes the error
        for col in numeric_feature_cols:
            try:
                 np.array(df[col], dtype=np.float32)
            except ValueError:
                 print(f"  Issue potentially in column: {col}. Contains non-numeric values or NaNs mixed with non-floats?")
                 print(f"  Sample values: {df[col].unique()[:10]}") # Show some unique values
        return None, None

    y = np.array(label_df).ravel()
    return x, y

def impute_nans_with_train_mean(X_train, X_test):
     """Imputes NaNs using column means calculated *only* from training data."""
     imputed_train = False
     try:
         col_mean_train = np.nanmean(X_train, axis=0)
     except TypeError:
         print("Error: Non-numeric data found in features during nanmean calculation. Check feature columns.")
         return None, None

     if np.isnan(X_train).any():
         imputed_train = True
         print("Warning: NaNs found in training features. Imputing with column mean.")
         inds = np.where(np.isnan(X_train))
         if np.isnan(col_mean_train[inds[1]]).any():
             print("Error: Cannot impute NaNs in training set - column mean is NaN for some columns with NaNs.")
             nan_mean_cols = np.where(np.isnan(col_mean_train))[0]
             problem_cols_indices = set(inds[1][np.isin(inds[1], nan_mean_cols)])
             print(f"Columns indices with NaNs where mean is also NaN: {problem_cols_indices}")
             # Optionally print feature names if FEATURE_COLS is accessible globally
             # print(f"Feature names: {[FEATURE_COLS[i] for i in problem_cols_indices]}")
             return None, None
         X_train[inds] = np.take(col_mean_train, inds[1])

     if np.isnan(X_test).any():
         print("Warning: NaNs found in test features. Imputing with training column mean.")
         col_mean_impute = col_mean_train
         inds_test = np.where(np.isnan(X_test))
         valid_means = np.take(col_mean_impute, inds_test[1])

         if np.isnan(valid_means).any():
             print("Error: Cannot impute NaNs in test set - corresponding training column mean is NaN.")
             nan_mean_cols_test = np.where(np.isnan(col_mean_impute))[0]
             problem_cols_indices_test = set(inds_test[1][np.isin(inds_test[1], nan_mean_cols_test)])
             print(f"Columns indices with NaNs where training mean is also NaN: {problem_cols_indices_test}")
             # Optionally print feature names if FEATURE_COLS is accessible globally
             # print(f"Feature names: {[FEATURE_COLS[i] for i in problem_cols_indices_test]}")
             return None, None
         X_test[inds_test] = valid_means

     return X_train, X_test


# --- Model Application Functions (Simplified: No GridSearchCV) ---

def apply_RFR(df_test, df_train, target_label, feature_cols):
    global args
    """Fits RFR with default parameters, predicts, returns results."""
    print("Applying RandomForestRegressor (Default Params)...")
    # 1. Split data
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None

    # 2. Handle NaN labels
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize result variables
    y_pred = None
    oob = None
    f_imp = None
    fit_time = np.nan

    # --- Define Default Parameters ---
    default_params = {
        'n_estimators': 160,
        'max_features': 0.4,
        'min_samples_leaf': args.temp_param,
     #   'random_state': 42,
        'n_jobs': -1,
     #   'oob_score': True # Calculate OOB score with default params
    }

    # --- Fit with Default Parameters ---
    print(f"  Using default RFR parameters: {default_params}")
    start_time = time.time()
    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"  Default RFR fitted in {fit_time:.2f} seconds.")
    y_pred = model.predict(X_test)
    f_imp = model.feature_importances_
    oob = model.oob_score_ if hasattr(model, 'oob_score_') else None
    metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                "R^2 Score": r2_score(y_test, y_pred),
                "OOB Score": oob,
                "Fit Time (s)": fit_time }
    log_experiment("Random Forest", default_params, metrics)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp


def apply_KNN(df_test, df_train, target_label, feature_cols):
    """Fits KNN with default parameters (in Pipeline), predicts, returns results."""
    print("Applying K Nearest Neighbors (Default Params)...")
    # 1. Split data
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None

    # 2. Handle NaN labels
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # KNN doesn't provide feature importances
    oob = None   # Not applicable
    fit_time = np.nan

    # --- Define Default Parameters for KNN step ---
    default_params = {
        'n_neighbors': 5,
     #   'weights': 'uniform',
     #   'metric': 'minkowski'
    }

    # --- Fit with Default Parameters using Pipeline ---
    print(f"  Using default KNN parameters within Pipeline: {default_params}")
    start_time = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(**default_params))
    ])
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"  Default KNN Pipeline fitted in {fit_time:.2f} seconds.")
    y_pred = pipe.predict(X_test)
    metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                "R^2 Score": r2_score(y_test, y_pred),
                "Fit Time (s)": fit_time }
    log_experiment("K Nearest Neighbors", default_params, metrics)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp


def apply_SVM(df_test, df_train, target_label, feature_cols):
    """Fits SVR with default parameters (in Pipeline), predicts, returns results."""
    print("Applying Support Vector Machine (SVR) (Default Params)...")
    # 1. Split data
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None

    # 2. Handle NaN labels
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # Not standard for kernel SVR
    oob = None   # Not applicable
    fit_time = np.nan

    # --- Define Default Parameters for SVR step ---
    default_params = {
     #   'kernel': 'rbf',
     #   'C': 1.0,
     #   'gamma': 'scale',
     #   'epsilon': 5e-6 # Keep fixed as per original script intention
    }

    # --- Fit with Default Parameters using Pipeline ---
    print(f"  Using default SVR parameters within Pipeline: {default_params}")
    start_time = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**default_params))
    ])
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"  Default SVR Pipeline fitted in {fit_time:.2f} seconds.")
    y_pred = pipe.predict(X_test)
    metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                "R^2 Score": r2_score(y_test, y_pred),
                "Fit Time (s)": fit_time }
    log_experiment("Support Vector Machine (SVR)", default_params, metrics)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp


def apply_BayesianRidge(df_test, df_train, target_label, feature_cols):
    """Fits BayesianRidge with default parameters (in Pipeline), predicts, returns results."""
    print("Applying Bayesian Ridge (Default Params)...")
    # 1. Split data
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None

    # 2. Handle NaN labels
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None
    oob = None   # Not applicable
    fit_time = np.nan

    # --- Define Default Parameters for BayesianRidge step ---
    default_params = {} # Use sklearn defaults

    # --- Fit with Default Parameters using Pipeline ---
    print("  Using default BayesianRidge parameters within Pipeline (Sklearn defaults).")
    start_time = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('br', BayesianRidge(**default_params))
    ])
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"  Default BayesianRidge Pipeline fitted in {fit_time:.2f} seconds.")
    y_pred = pipe.predict(X_test)
    f_imp = np.abs(pipe.named_steps['br'].coef_)
    metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                "R^2 Score": r2_score(y_test, y_pred),
                "Fit Time (s)": fit_time }
    log_experiment("Bayesian Ridge", default_params if default_params else "Sklearn Defaults", metrics)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp


def apply_Lasso(df_test, df_train, target_label, feature_cols):
    """Fits Lasso with default parameters (in Pipeline), predicts, returns results."""
    print("Applying Lasso Regression (Default Params)...")
    # 1. Split data
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None

    # 2. Handle NaN labels
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None
    oob = None   # Not applicable
    fit_time = np.nan

    # --- Define Default Parameters for Lasso step ---
    default_params = {
        #'alpha': 1.0,
        # 'max_iter': 5000, # Consider increasing if convergence issues arise
        #'random_state': 42,
        #'tol': 1e-4
    }

    # --- Fit with Default Parameters using Pipeline ---
    print(f"  Using default Lasso parameters within Pipeline: {default_params}")
    start_time = time.time()
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(**default_params))
    ])
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"  Default Lasso Pipeline fitted in {fit_time:.2f} seconds.")
    y_pred = pipe.predict(X_test)
    f_imp = np.abs(pipe.named_steps['lasso'].coef_)
    metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                "R^2 Score": r2_score(y_test, y_pred),
                "Fit Time (s)": fit_time }
    log_experiment("Lasso Regression", default_params, metrics)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp


# --- Data Processing Functions (fit_transformation, truncate unchanged) ---

def fit_transformation(df, target_label, baseline_ll_col, group_id_col, trans=False):
    """Transforms target label based on baseline LL."""
    print("Applying target transformation...")
    if baseline_ll_col not in df.columns or target_label not in df.columns:
        print(f"Error: Required columns for transformation missing ({baseline_ll_col}, {target_label}). Skipping.")
        return df

    df.loc[:, target_label] = pd.to_numeric(df[target_label], errors='coerce')
    df.loc[:, baseline_ll_col] = pd.to_numeric(df[baseline_ll_col], errors='coerce')

    mask = pd.notna(df[target_label]) & pd.notna(df[baseline_ll_col]) & (df[baseline_ll_col] != 0)
    df.loc[mask, target_label] = df.loc[mask, target_label] / -df.loc[mask, baseline_ll_col]
    df.loc[~mask, target_label] = np.nan

    if trans:
        print("Applying exp2 transformation...")
        mask_trans = pd.notna(df[target_label])
        try:
            # Apply exp2 safely - consider clipping if necessary
            # clipped_values = np.clip(df.loc[mask_trans, target_label] + 1, -50, 50)
            # df.loc[mask_trans, target_label] = np.exp2(clipped_values)
            df.loc[mask_trans, target_label] = np.exp2(df.loc[mask_trans, target_label] + 1)
        except (OverflowError, ValueError) as e:
            print(f"Warning: Error during exp2 transformation: {e}. Setting problematic values to NaN.")
            df.loc[mask_trans, target_label] = np.nan

    print("Transformation finished.")
    return df

def truncate(df, group_id_col, kfold):
    """Truncates DataFrame groups to be divisible by kfold for CV splitting."""
    print("Truncating datasets for K-Fold/LOO...")
    if group_id_col not in df.columns:
        print(f"Error: Group ID column '{group_id_col}' not found for truncation.")
        return df, np.array([]), 0

    df = df.dropna(subset=[group_id_col])
    groups_ids = df[group_id_col].unique()
    n_groups = len(groups_ids)
    print(f"Initial number of dataset groups: {n_groups}")

    if n_groups == 0:
        print("Warning: No groups found after dropping NaNs in group ID column.")
        return df, groups_ids, 0

    if isinstance(kfold, str) and kfold.upper() == "LOO":
        kfold_num = n_groups
        if kfold_num <= 0:
             print("Warning: Cannot perform LOO with 0 groups.")
             return df.reset_index(drop=True), groups_ids, 0
        print("Using Leave-One-Out (LOO) CV strategy.")
        test_batch_size = 1
        print(f"Number of dataset groups (folds): {n_groups}")
        print(f"Test batch size per fold: {test_batch_size}")
        return df.reset_index(drop=True), groups_ids, test_batch_size

    elif isinstance(kfold, int):
        kfold_num = kfold
        if kfold_num <= 0:
            print("Error: KFOLD value must be a positive integer or 'LOO'.")
            return df.reset_index(drop=True), groups_ids, 0

        if n_groups < kfold_num:
            print(f"Warning: Number of groups ({n_groups}) is less than KFOLD ({kfold_num}). Setting KFOLD to {n_groups}.")
            kfold_num = n_groups

        if kfold_num <= 0:
             print("Warning: Cannot perform CV with 0 effective folds.")
             return df.reset_index(drop=True), groups_ids, 0

        ndel = n_groups % kfold_num
        if ndel != 0:
            print(f"Removing {ndel} groups to make total ({n_groups}) divisible by KFOLD={kfold_num}.")
            ids_to_remove = groups_ids[-ndel:]
            df = df[~df[group_id_col].isin(ids_to_remove)].copy()
            groups_ids = df[group_id_col].unique()

        final_n_groups = len(groups_ids)
        if final_n_groups == 0 or kfold_num == 0:
            print("Warning: No groups remaining after truncation or kfold is zero.")
            test_batch_size = 0
        else:
            test_batch_size = final_n_groups // kfold_num

        print(f"Number of dataset groups after truncation: {final_n_groups}")
        print(f"Test batch size per fold: {test_batch_size}")
        return df.reset_index(drop=True), groups_ids, test_batch_size
    else:
        print(f"Error: Invalid kfold type: {type(kfold)}. Must be int or 'LOO'.")
        return df, groups_ids, 0

# --- Cross-Validation / Validation Set Function (Simplified) ---

def evaluate_model(df, target_label, feature_cols, group_id_col, model_name, kfold_value,
                   validation_set=False, validation_set_path=None,
                   trans=False, random_choice=False, scale_score=False):
    """Performs K-Fold CV or evaluates on a validation set using default model parameters."""
    res_dict = {}
    oobs, f_imps = [], []
    df_out = df.copy()

    model_func_map = {
        'RFR': apply_RFR,
        'KNN': apply_KNN,
        'SVM': apply_SVM,
        'BR': apply_BayesianRidge,
        'Lasso': apply_Lasso
    }
    apply_func = model_func_map.get(model_name)
    if not apply_func:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(model_func_map.keys())}")
    print(f"Selected model function: {apply_func.__name__}")

    if not validation_set:
        # --- K-Fold Cross Validation ---
        print(f"\nStarting {kfold_value}-Fold Cross Validation (Outer Loop)...")
        df_cv, groups_ids, test_batch_size = truncate(df_out, group_id_col, kfold_value)

        if test_batch_size <= 0 or len(groups_ids) == 0:
             print(f"Warning: Cannot perform CV. test_batch_size={test_batch_size}, num_groups={len(groups_ids)}. Check data and kfold value.")
             if 'pred' not in df_out.columns: df_out["pred"] = np.nan
             return {}, df_out

        actual_k = math.ceil(len(groups_ids) / test_batch_size) if test_batch_size > 0 else 0
        print(f"Actual number of outer folds to run: {actual_k}")

        df_cv["pred"] = np.nan
        fold_num = 0
        for low_i_idx in range(0, len(groups_ids), test_batch_size):
            fold_num += 1
            up_i_idx = low_i_idx + test_batch_size
            test_group_ids = groups_ids[low_i_idx:up_i_idx]
            train_group_ids = np.setdiff1d(groups_ids, test_group_ids)

            print(f"\n--- Outer Fold {fold_num}/{actual_k} ---")
            test_ids_str = str(list(test_group_ids[:5])) + ('...' if len(test_group_ids) > 5 else '')
            print(f"  Test Group IDs: {test_ids_str}")

            df_test_fold = df_cv[df_cv[group_id_col].isin(test_group_ids)].copy()
            df_train_fold = df_cv[df_cv[group_id_col].isin(train_group_ids)].copy()

            if df_train_fold.empty or df_test_fold.empty:
                print(f"Warning: Fold {fold_num} has empty train ({len(df_train_fold)}) or test ({len(df_test_fold)}) set. Skipping.")
                continue

            print(f"  Training set size: {len(df_train_fold)}, Test set size: {len(df_test_fold)}")

            # Call the selected model function (now simplified, no grid search flag)
            y_pred_fold, oob_fold, f_imp_fold = apply_func(
                df_test_fold, df_train_fold, target_label, feature_cols
            )

            if oob_fold is not None: oobs.append(oob_fold)
            if f_imp_fold is not None: f_imps.append(f_imp_fold)
            if y_pred_fold is not None:
                if len(y_pred_fold) == len(df_test_fold.index):
                     df_cv.loc[df_test_fold.index, "pred"] = y_pred_fold
                else:
                     print(f"Warning: Prediction length mismatch in fold {fold_num}. Cannot assign predictions.")
                     # df_cv.loc[df_test_fold.index, "pred"] = np.nan

        print("\nCross Validation Finished.")
        df_out = df_cv

    else:
        # --- Validation Set Mode ---
        print("\nUsing Validation Set strategy...")
        if not validation_set_path or not os.path.exists(validation_set_path):
            print(f"Error: Validation set path missing or not found: {validation_set_path}")
            if 'pred' not in df_out.columns: df_out["pred"] = np.nan
            return {}, df_out

        df_train = df_out
        print(f"Training data size: {len(df_train)}")

        try:
            print(f"Loading validation set from: {validation_set_path}")
            df_test = pd.read_csv(validation_set_path, dtype=types_dict)
            print(f"Validation set size: {len(df_test)}")
        except Exception as e:
            print(f"Error reading validation set CSV: {e}")
            if 'pred' not in df_out.columns: df_out["pred"] = np.nan
            return {}, df_out

        required_val_cols = feature_cols + [target_label, group_id_col]
        if trans: required_val_cols.append(BASELINE_LL_COL) # Use updated baseline col name
        missing_val_cols = [col for col in required_val_cols if col not in df_test.columns]
        if missing_val_cols:
            print(f"Error: Validation set CSV missing required columns: {missing_val_cols}")
            if 'pred' not in df_test.columns: df_test["pred"] = np.nan
            return {}, df_test

        if trans:
            print("Applying transformation to validation set...")
            # Use updated baseline col name here
            df_test = fit_transformation(df_test.copy(), target_label, BASELINE_LL_COL, group_id_col, trans=True)
        else:
            df_test.loc[:, target_label] = pd.to_numeric(df_test[target_label], errors='coerce')

        initial_val_rows = len(df_test)
        df_test = df_test.dropna(subset=[target_label])
        if len(df_test) < initial_val_rows:
            print(f"Dropped {initial_val_rows - len(df_test)} rows with NaN target label from validation set.")

        if df_train.empty or df_test.empty:
            print(f"Error: Training ({len(df_train)}) or validation ({len(df_test)}) set is empty after preprocessing.")
            if 'pred' not in df_test.columns: df_test["pred"] = np.nan
            df_out = df_test
            return {}, df_out

        print(f"Final training size: {len(df_train)}, Final validation size: {len(df_test)}")

        # Call the selected model function
        y_pred_val, oob_val, f_imp_val = apply_func(
            df_test, df_train, target_label, feature_cols
        )

        if oob_val is not None: oobs.append(oob_val)
        if f_imp_val is not None: f_imps.append(f_imp_val)
        if y_pred_val is not None:
             if len(y_pred_val) == len(df_test.index):
                 df_test["pred"] = y_pred_val
             else:
                 print(f"Warning: Prediction length mismatch in validation set. Cannot assign predictions.")
                 df_test["pred"] = np.nan
        else:
            df_test["pred"] = np.nan

        df_out = df_test

    # --- Calculate final scores ---
    print("\nCalculating final scores...")
    if 'pred' not in df_out.columns or df_out['pred'].isnull().all():
        print("Warning: 'pred' column missing or contains only NaNs in the final dataframe. Cannot calculate scores.")
        res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan
        valid_f_imps = [fi for fi in f_imps if fi is not None]
        res_dict['f_importance'] = np.nanmean(np.array(valid_f_imps), axis=0) if valid_f_imps else None
        res_dict["rank_first_pred"] = {}
        res_dict["rank_first_true"] = {}
        res_dict["spearman_corr"] = []
        return res_dict, df_out

    df_scored = df_out.dropna(subset=['pred', target_label])
    if df_scored.empty:
        print("Warning: No valid rows remaining after dropping NaNs in 'pred' or target label. Scores will be NaN.")
        res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan
        valid_f_imps = [fi for fi in f_imps if fi is not None]
        res_dict['f_importance'] = np.nanmean(np.array(valid_f_imps), axis=0) if valid_f_imps else None
        res_dict["rank_first_pred"] = {}
        res_dict["rank_first_true"] = {}
        res_dict["spearman_corr"] = []
    else:
        print(f"Scoring based on {len(df_scored)} rows with valid predictions and labels.")
        rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df_scored, target_label, group_id_col, random_choice, scale_score)
        res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan
        valid_f_imps = [fi for fi in f_imps if fi is not None]
        res_dict['f_importance'] = np.nanmean(np.array(valid_f_imps), axis=0) if valid_f_imps else None
        res_dict["rank_first_pred"] = rank_pred_by_ds
        res_dict["rank_first_true"] = rank_test_by_ds
        res_dict["spearman_corr"] = corrs

    return res_dict, df_out


# --- Result Formatting (print_and_index_results unchanged) ---

def print_and_index_results(res_dict, features):
    """Prints aggregated results and creates a summary DataFrame."""
    df_agg_scores = pd.DataFrame(index=[0])

    spearman_corrs = res_dict.get('spearman_corr', [])
    valid_corrs = [c for c in spearman_corrs if pd.notna(c)]
    mean_corr = np.mean(valid_corrs) if valid_corrs else np.nan
    median_corr = np.median(valid_corrs) if valid_corrs else np.nan
    df_agg_scores['mean_spearman_corr'] = mean_corr
    df_agg_scores['median_spearman_corr'] = median_corr
    print(f"\nSpearman Correlation (across {len(valid_corrs)} groups with valid scores): Mean={mean_corr:.4f}, Median={median_corr:.4f}")

    ranks_pred_raw = res_dict.get('rank_first_pred', {}).values()
    ranks_true_raw = res_dict.get('rank_first_true', {}).values()
    ranks_pred = [r for r in ranks_pred_raw if pd.notna(r)]
    ranks_true = [r for r in ranks_true_raw if pd.notna(r)]
    mean_rank_pred = np.mean(ranks_pred) if ranks_pred else np.nan
    median_rank_pred = np.median(ranks_pred) if ranks_pred else np.nan
    mean_rank_true = np.mean(ranks_true) if ranks_true else np.nan
    median_rank_true = np.median(ranks_true) if ranks_true else np.nan
    df_agg_scores['mean_rank_best_pred'] = mean_rank_pred
    df_agg_scores['median_rank_best_pred'] = median_rank_pred
    df_agg_scores['mean_rank_best_true'] = mean_rank_true
    df_agg_scores['median_rank_best_true'] = median_rank_true
    print(f"Best Predicted Rank (%): Mean={mean_rank_pred:.2f}%, Median={median_rank_pred:.2f}% (over {len(ranks_pred)} groups)")
    print(f"Best True Rank (%): Mean={mean_rank_true:.2f}%, Median={median_rank_true:.2f}% (over {len(ranks_true)} groups)")

    oob_score = res_dict.get('oob', np.nan)
    df_agg_scores['mean_oob_score'] = oob_score
    print(f"Mean OOB Score: {'N/A' if pd.isna(oob_score) else f'{oob_score:.4f}'}")

    mean_importances = res_dict.get('f_importance')
    print("\nFeature Importances (averaged over folds/runs):")
    if mean_importances is not None:
        num_expected_features = len(features)
        if len(mean_importances) == num_expected_features:
            feature_importance_pairs = sorted(zip(features, mean_importances), key=lambda x: x[1], reverse=True)
            for feature, imp_val in feature_importance_pairs:
                 colname = "imp_" + feature
                 df_agg_scores[colname] = imp_val
                 print(f"  {feature}: {imp_val:.4f}")
        else:
            print(f"  Warning: Mismatch between number of importances ({len(mean_importances)}) and expected features ({num_expected_features}). Cannot reliably assign names.")
            for i, imp_val in enumerate(mean_importances):
                 colname = f"imp_raw_{i}"
                 df_agg_scores[colname] = imp_val
    else:
        print("  Not available for this model or run.")

    num_groups_scored = len(valid_corrs)
    print(f"\nNumber of groups processed for rank/correlation scores: {num_groups_scored}")
    return df_agg_scores

# --- New function to process kfold argument (unchanged) ---
def process_kfold_arg(kfold_input_string):
    """Processes the kfold command line argument string."""
    if isinstance(kfold_input_string, str) and kfold_input_string.upper() == "LOO":
        return "LOO"
    else:
        try:
            kfold_val = int(kfold_input_string)
            if kfold_val <= 0:
                 print("Error: KFOLD value must be positive or 'LOO'.")
                 return None
            return kfold_val
        except (ValueError, TypeError):
             print(f"Error: Invalid KFOLD value: {kfold_input_string}. Must be a positive integer or 'LOO'.")
             return None

# --- Main Execution Block (Simplified Args) ---

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run ML algorithm using default parameters on TBR features.')
    parser.add_argument('--input_features_dir', '-i', required=True, help=f'Path to the directory containing feature CSV files (e.g., matching "{INPUT_FILENAME_PATTERN}"). Searches recursively.')
    parser.add_argument('--output_scores_csv', '-s', required=True, help='Path to save aggregated scores CSV.')
    parser.add_argument('--output_preds_csv', '-p', required=True, help='Path to save predictions CSV (includes original data + pred column).')
    parser.add_argument('--model', '-m', type=str, default='RFR', choices=['RFR', 'KNN', 'SVM', 'BR', 'Lasso'], help='ML model to use.')
    parser.add_argument('--kfold', '-k', type=str, default=str(KFOLD), help=f'Number of folds for Outer CV or "LOO" (Leave-One-Out) (default: {KFOLD}).')
    parser.add_argument('--transform_target', '-trans', default=False, action='store_true', help=f'Apply transformation to target (division by {BASELINE_LL_COL} and optional exp2).')
    parser.add_argument('--validation_set', '-val', default=False, action='store_true', help='Use validation set mode instead of K-Fold CV.')
    parser.add_argument('--validation_set_path', '-valpath', type=str, default=None, help='Path to validation set CSV (required if --validation_set is used).')
    parser.add_argument('--temp_param', '-t', type=int, default=0.5)

    args = parser.parse_args()

    # --- Initial Setup & Logging ---
    print("--- Script Start ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Selected: {args.model}")
    # print(f"GridSearchCV Enabled: False") # Grid search is always disabled now
    print(f"Target Transformation Enabled: {args.transform_target}")
    print(f"Evaluation Mode: {'Validation Set' if args.validation_set else 'Cross-Validation'}")

    if args.validation_set:
        if not args.validation_set_path:
            print("Error: --validation_set_path is required when --validation_set is used.")
            sys.exit(1)
        if not os.path.exists(args.validation_set_path):
            print(f"Error: Validation file not found: {args.validation_set_path}")
            sys.exit(1)
        print(f"Validation Set Path: {args.validation_set_path}")
        kfold_val = None
    else:
        kfold_val = process_kfold_arg(args.kfold)
        if kfold_val is None:
            sys.exit(1)
        print(f"Outer CV Folds: {kfold_val}")

    # --- Load and Merge Data ---
    input_dir = args.input_features_dir
    print(f"\nLoading and merging feature data from: {input_dir}")
    if not os.path.isdir(input_dir):
        print(f"Error: Input features directory not found: {input_dir}")
        sys.exit(1)

    # Use the INPUT_FILENAME_PATTERN defined earlier
    search_pattern = os.path.join(input_dir, '**', INPUT_FILENAME_PATTERN)
    print(f"Searching pattern: {search_pattern}")
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        print(f"Error: No files matching '{INPUT_FILENAME_PATTERN}' found recursively in directory: {input_dir}")
        sys.exit(1)
    print(f"Found {len(all_files)} feature files to merge.")

    df_list = []
    read_errors = 0
    for f in all_files:
        try:
            df_temp = pd.read_csv(f, dtype=types_dict)
            df_list.append(df_temp)
        except Exception as e:
            print(f"Warning: Error reading file {f}: {e}. Skipping file.")
            read_errors += 1

    if not df_list:
        print("Error: No data loaded. Check input files and directory structure.")
        sys.exit(1)
    if read_errors > 0:
         print(f"Warning: Skipped {read_errors} files due to read errors or missing columns.")

    df_learning = pd.concat(df_list, ignore_index=True)
    print(f"Merged data into DataFrame with {len(df_learning)} rows and {len(df_learning.columns)} columns.")

    # --- Preprocessing ---
    print("\nPreprocessing data...")
    # Verify required columns using updated BASELINE_LL_COL
    required_cols = FEATURE_COLS + [LABEL, GROUP_ID_COL, BASELINE_LL_COL]
    missing_cols = [col for col in required_cols if col not in df_learning.columns]
    if missing_cols:
        print(f"Error: Merged CSV missing required columns: {missing_cols}")
        sys.exit(1)

    if args.transform_target:
        if args.validation_set:
            print("Note: Transformation will be applied to the training data now.")
            print("      It will also be applied separately to the validation set before prediction.")
        df_learning = fit_transformation(df_learning, LABEL, BASELINE_LL_COL, GROUP_ID_COL, trans=True)
    else:
        df_learning[LABEL] = pd.to_numeric(df_learning[LABEL], errors='coerce')

    initial_rows = len(df_learning)
    df_learning = df_learning.dropna(subset=[LABEL])
    rows_dropped = initial_rows - len(df_learning)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with NaN target label ('{LABEL}').")

    if df_learning.empty:
        print("Error: No valid data remaining after preprocessing. Cannot train model.")
        sys.exit(1)

    # --- Run Evaluation ---
    print("\nStarting model evaluation (using default parameters)...")

    start_time = time.time()
    res_dict, df_with_preds = evaluate_model(
        df=df_learning,
        target_label=LABEL,
        feature_cols=FEATURE_COLS,
        group_id_col=GROUP_ID_COL,
        model_name=args.model,
        kfold_value=kfold_val,
        validation_set=args.validation_set,
        validation_set_path=args.validation_set_path,
        trans=args.transform_target,
        random_choice=False, # Use max prediction for rank
        scale_score=True, # Ranks as %
        #nEstimators=args.nEstimators
    )
    print(f"\nEvaluation total time: {time.time() - start_time:.2f} seconds")

    # --- Save Predictions ---
    output_preds_path = args.output_preds_csv
    print(f"\nSaving DataFrame with predictions to: {output_preds_path}")
    try:
        os.makedirs(os.path.dirname(output_preds_path), exist_ok=True)
        if 'pred' not in df_with_preds.columns:
            print("Warning: 'pred' column not found in the final DataFrame. Saving without predictions.")
        df_with_preds.to_csv(output_preds_path, index=False, float_format='%.6f')
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions CSV to {output_preds_path}: {e}")

    # --- Print and Save Aggregated Scores ---
    print("\n--- Aggregated Results ---")
    df_scores = print_and_index_results(res_dict, FEATURE_COLS)

    output_scores_path = args.output_scores_csv
    print(f"\nSaving aggregated scores to: {output_scores_path}")
    try:
        os.makedirs(os.path.dirname(output_scores_path), exist_ok=True)
        df_scores.to_csv(output_scores_path, index=False, float_format='%.6f')
        print("Aggregated scores saved successfully.")
    except Exception as e:
        print(f"Error saving scores CSV to {output_scores_path}: {e}")

    # --- Script End ---
    print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--- Script End ---")