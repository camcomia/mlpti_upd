#########################################################################
##           Copyright (C). All Rights Reserved.                       ##
##   "Harnessing machine learning to guide                             ##
##                    phylogenetic-tree search algorithms"             ##
##                                                                     ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                     ##
## Adapted script for learning algorithm, designed to work with combined ##
## feature data generated from the NNI->SPR pipeline.                  ##
#########################################################################

# Standard libraries
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
from typing import List, Optional

# Required ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso, BayesianRidge
    from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.inspection import permutation_importance # Keep for potential future use (requires fitted model)
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV
    # Import Pipeline
    from sklearn.pipeline import Pipeline
except ImportError as e:
    print(f"Error: Missing scikit-learn component ({e}).")
    print("Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

# --- Constants ---
LABEL = "d_ll"
FEATURES = OrderedDict([
    ("bl", "edge_length"), ("longest", "longest_branch"),
    ("tbl_p","tbl_pruned"),("tbl_r","tbl_remaining"),
    ("longest_p","longest_pruned"),("longest_r","longest_remaining"),
    ("ntaxa_p", "num_taxa_prune"), ("ntaxa_r", "num_taxa_rgft"),
    ("bl_rgft", "edge_length_rgftSite"),
    ("ntaxa_p_rgft", "num_taxa_prune_rgftSite"),
    ("ntaxa_r_rgft", "num_taxa_rgft_rgftSite"),
    ("tbl_p_rgft", "tbl_pruned_rgftSite"),
    ("tbl_r_rgft", "tbl_remaining_rgftSite"),
    ("longest_p_rgft", "longest_pruned_rgftSite"),
    ("longest_r_rgft", "longest_remaining_rgftSite"),
    ("top_dist","topology_dist_between"), ("bl_dist","tbl_dist_between"),
    ("res_bl", "res_tree_edge_length"),
    ("group_id", "orig_ds_id") # Maps internal key 'group_id' to CSV column name 'orig_ds_id'
])
FEATURE_COLS = [col for key, col in FEATURES.items() if key != "group_id"] # Get actual column names, excluding the group ID
KFOLD = 10 # Default for outer CV if not LOO
GROUP_ID_COL = FEATURES.get("group_id", "orig_ds_id") # Get the actual column name for grouping
ORIG_DS_LL_COL = "orig_ds_ll"
NNI_TREE_LL_COL = "nni_tree_ll"
INNER_CV_FOLDS = 10 # Standardized inner folds for GridSearchCV

# Define column types for efficient reading with pandas
list_str = ['prune_name', 'rgft_name', GROUP_ID_COL]
list_int = [] # Define integer columns if any
list_float = ['ll', NNI_TREE_LL_COL, ORIG_DS_LL_COL, LABEL] + FEATURE_COLS # Target label and all feature columns
types_dict = {}
for e in list_str: types_dict[e] = np.object_ # Use object for strings
for e in list_int: types_dict[e] = np.int32
for e in list_float: types_dict[e] = np.float32 # Use float32 for potentially large datasets


# --- Helper Functions ---

def log_experiment(model_name, params, metrics, best_params=None, grid_searched=True):
    """Logs experiment parameters and metrics."""
    search_type = "(GridSearchCV)" if grid_searched else "(Default Params)"
    print(f"--- {model_name} {search_type} Experiment ---")
    # If default params used, 'params' contains the defaults dictionary
    if not grid_searched and params:
        print(f"  Default Parameters Used: {params}")
    # If grid search used, 'params' might contain fixed params (like random_state)
    elif grid_searched and params:
         print(f"  Fixed Parameters (During GridSearch): {params}")

    if grid_searched and best_params:
        print(f"  Best Params (GridSearchCV): {best_params}")

    # Format metrics for printing
    printable_metrics = {k: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics.items() if v is not None}
    print(f"  Metrics: {printable_metrics}")
    print("-" * (len(model_name) + len(search_type) + 20))

def score_rank(df_by_ds, sortby, locatein, random_choice, scale_score):
    """Finds rank of the best predicted item within the true ranking.

    Args:
        df_by_ds (pd.DataFrame): DataFrame for a single dataset group.
        sortby (str): Column name to sort by to find the 'best' predicted item (e.g., 'pred').
        locatein (str): Column name representing the true ranking (e.g., target label 'd_ll').
        random_choice (bool): If True, randomly select an item instead of the max predicted.
        scale_score (bool): If True, scale the rank to a percentage (0-100).

    Returns:
        float: The rank (or scaled rank) of the best predicted item, or np.nan if error.
    """
    if df_by_ds.empty or sortby not in df_by_ds or locatein not in df_by_ds: return np.nan

    # Ensure columns are numeric, handle potential errors during conversion
    df_by_ds.loc[:, sortby] = pd.to_numeric(df_by_ds[sortby], errors='coerce')
    df_by_ds.loc[:, locatein] = pd.to_numeric(df_by_ds[locatein], errors='coerce')

    # Drop rows where sorting or locating columns became NaN after conversion
    df_by_ds = df_by_ds.dropna(subset=[sortby, locatein])
    if df_by_ds.empty: return np.nan

    # Find the index of the best predicted item (or random)
    if random_choice:
        best_pred_ix = np.random.choice(df_by_ds.index, 1, replace=False)[0]
    else:
        try:
            best_pred_ix = df_by_ds[sortby].idxmax() # Find index of max value in 'sortby' column
        except ValueError: # Handle cases where idxmax fails (e.g., all NaNs)
            return np.nan

    # Sort by the true ranking ('locatein') to find the rank
    temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index() # Reset index to get rank easily

    # Find the 0-based rank of the best predicted item in the true sorted list
    rank_list = temp_df.index[temp_df["index"] == best_pred_ix].tolist() # Find where original index appears
    if not rank_list: return np.nan # Should not happen if best_pred_ix was valid
    best_pred_rank = min(rank_list) + 1 # Add 1 to get 1-based rank

    # Scale the rank if requested
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
        df_by_ds = df_by_ds_orig.copy() # Avoid SettingWithCopyWarning

        # Calculate rank of best predicted item in true ranking
        rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", target_label, random_choice, scale_score)
        # Calculate rank of best true item in predicted ranking
        rank_test_by_ds[group_id] = score_rank(df_by_ds, target_label, "pred", random_choice, scale_score)

        # Calculate Spearman correlation for the group
        temp_df = df_by_ds[[target_label, "pred"]].dropna()
        if len(temp_df) > 1 and temp_df[target_label].nunique() > 1 and temp_df["pred"].nunique() > 1:
            try:
                # Calculate Spearman correlation between true label and prediction
                corr = temp_df.corr(method='spearman').iloc[0, 1]
                sp_corrs.append(corr)
            except (IndexError, ValueError): # Handle potential errors during correlation calculation
                 sp_corrs.append(np.nan)
        else: # Not enough data or variance for correlation
            sp_corrs.append(np.nan)

    return rank_pred_by_ds, rank_test_by_ds, sp_corrs

def split_features_label(df, target_label, feature_cols):
    """Splits DataFrame into features (X) and labels (y)."""
    # Check if all specified feature columns exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in DataFrame: {missing_features}")
        return None, None
    # Check if the target label column exists
    if target_label not in df.columns:
        print(f"Error: Target label '{target_label}' not found in DataFrame.")
        return None, None

    # Select only numeric feature columns to avoid errors when converting to numpy
    numeric_feature_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_feature_cols) != len(feature_cols):
        non_numeric = set(feature_cols) - set(numeric_feature_cols)
        print(f"Warning: Non-numeric columns found in specified features: {non_numeric}. These will be excluded.")

    # Create feature matrix (X) and label vector (y)
    attributes_df = df[numeric_feature_cols]
    label_df = df[target_label]
    x = np.array(attributes_df)
    y = np.array(label_df).ravel() # Ensure y is a 1D array

    return x, y

def impute_nans_with_train_mean(X_train, X_test):
     """Imputes NaNs using column means calculated *only* from training data."""
     imputed_train = False # Flag to track if training data was imputed
     try:
         # Calculate mean for each column in the training set, ignoring NaNs
         col_mean_train = np.nanmean(X_train, axis=0)
     except TypeError:
         print("Error: Non-numeric data found in features during nanmean calculation. Check feature columns.")
         return None, None

     # Check if training data has any NaNs
     if np.isnan(X_train).any():
         imputed_train = True
         print("Warning: NaNs found in training features. Imputing with column mean.")
         # Find indices (row, col) of NaNs in training data
         inds = np.where(np.isnan(X_train))
         # Check if any column mean needed for imputation is itself NaN (e.g., column was all NaNs)
         if np.isnan(col_mean_train[inds[1]]).any():
             print("Error: Cannot impute NaNs in training set - column mean is NaN for some columns with NaNs.")
             # Identify problematic columns
             nan_mean_cols = np.where(np.isnan(col_mean_train))[0]
             problem_cols_indices = set(inds[1][np.isin(inds[1], nan_mean_cols)])
             print(f"Columns indices with NaNs where mean is also NaN: {problem_cols_indices}")
             # Optionally print feature names if available: print(f"Feature names: {[FEATURE_COLS[i] for i in problem_cols_indices]}")
             return None, None # Cannot proceed with imputation
         # Impute NaNs using the calculated means for the respective columns
         X_train[inds] = np.take(col_mean_train, inds[1])

     # Check if test data has any NaNs
     if np.isnan(X_test).any():
         print("Warning: NaNs found in test features. Imputing with training column mean.")
         # Use means calculated *before* any potential imputation of the training set
         col_mean_impute = col_mean_train # Means are based purely on original training data
         # Find indices (row, col) of NaNs in test data
         inds_test = np.where(np.isnan(X_test))
         # Get the corresponding training means for the columns with NaNs in the test set
         valid_means = np.take(col_mean_impute, inds_test[1])

         # Check if any required training mean is NaN
         if np.isnan(valid_means).any():
             print("Error: Cannot impute NaNs in test set - corresponding training column mean is NaN.")
             # Identify problematic columns
             nan_mean_cols_test = np.where(np.isnan(col_mean_impute))[0]
             problem_cols_indices_test = set(inds_test[1][np.isin(inds_test[1], nan_mean_cols_test)])
             print(f"Columns indices with NaNs where training mean is also NaN: {problem_cols_indices_test}")
             # Optionally print feature names if available: print(f"Feature names: {[FEATURE_COLS[i] for i in problem_cols_indices_test]}")
             return None, None # Cannot proceed
         # Impute NaNs in test set using the valid training means
         X_test[inds_test] = valid_means

     return X_train, X_test


# --- Model Application Functions (Conditional GridSearchCV with default_params dict) ---

def apply_RFR(df_test, df_train, target_label, feature_cols, add_params, add_mf, no_gridsearch=False):
    """Performs GridSearchCV (or default fit) for RFR, predicts, returns results."""
    print(f"Applying RandomForestRegressor {'(Default Params)' if no_gridsearch else '(GridSearchCV)'}...")
    # 1. Split data into features (X) and labels (y)
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None # Error during split

    # 2. Handle potential NaN labels (cannot train/evaluate on these)
    train_valid_idx = ~np.isnan(y_train) # Boolean mask for valid training labels
    test_valid_idx = ~np.isnan(y_test)   # Boolean mask for valid test labels
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]

    # 3. Impute NaN features using training set means
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None or X_test is None: return None, None, None # Error during imputation

    # Check if data remains after filtering NaNs
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize result variables
    y_pred = None
    oob = None # Out-of-bag score (only relevant for RFR default)
    f_imp = None # Feature importances
    best_params = None # Best parameters from GridSearchCV
    best_score = np.nan # Best score from GridSearchCV (inner CV)
    search_time = np.nan # Time taken for fitting/searching

    # --- Define Default Parameters ---
    default_params = {
        'n_estimators': 100, # Number of trees in the forest
        'max_features': 0.4,   # Proportion of features to consider at each split
        'random_state': 42,     # Ensures reproducibility
        'n_jobs': -1,           # Use all available CPU cores for fitting default model
        'oob_score': True       # Calculate out-of-bag score (only if using default)
    }

    if no_gridsearch:
        # --- Fit with Default Parameters ---
        print("  Skipping GridSearchCV. Using default RFR parameters.")
        start_time = time.time()
        # Instantiate model using the default_params dictionary
        model = RandomForestRegressor(**default_params)
        model.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  Default RFR fitted in {search_time:.2f} seconds.")
        y_pred = model.predict(X_test)
        f_imp = model.feature_importances_ # Get feature importances
        oob = model.oob_score_ if hasattr(model, 'oob_score_') else None # Get OOB score
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "OOB Score": oob,
                    "Fit Time (s)": search_time }
        # Pass the used default_params to logger
        log_experiment("Random Forest", default_params, metrics, grid_searched=False)

    else:
        # --- Perform GridSearchCV ---
        # Define parameters specific to the base estimator within GridSearchCV
        rfr_grid_base_params = {
             'random_state': 42,
             'n_jobs': 1,       # Crucial: Base estimator uses 1 job, GridSearchCV handles parallelization
             'oob_score': False # OOB score is not reliable with inner CV splits
        }
        # Define the parameter grid to search over
        param_grid = {
            'n_estimators': [50, 70],
            'max_features': [0.4, 0.5]
        }


        print(f"  Parameter Grid for RFR GridSearchCV:\n  {param_grid}")
        rfr = RandomForestRegressor(**rfr_grid_base_params) # Base model instance
        start_time = time.time()
        # Setup GridSearchCV
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid,
                                   cv=INNER_CV_FOLDS, # Use predefined inner CV folds
                                   scoring='neg_mean_squared_error', # Score to optimize (negative MSE)
                                   n_jobs=4, # Use all cores for grid search itself
                                   verbose=1) # Print progress updates
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  GridSearchCV finished in {search_time:.2f} seconds.")

        # Extract results from grid search
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_ # Best score on inner CV (neg MSE)
        y_pred = best_model.predict(X_test) # Predict on the outer test set
        f_imp = best_model.feature_importances_
        oob = None # OOB score wasn't calculated during grid search
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Best Inner CV Score (Neg MSE)": best_score,
                    "GridSearch Time (s)": search_time }
        # Pass fixed base params used during grid search
        log_experiment("Random Forest", rfr_grid_base_params, metrics, best_params=best_params, grid_searched=True)

    # 4. Prepare full prediction array including NaNs for original indices
    # This ensures the output aligns with the original input DataFrame rows
    full_y_pred = np.full(len(test_valid_idx), np.nan) # Create array of NaNs with original length
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred # Fill in predictions at the valid indices

    return full_y_pred, oob, f_imp # Return predictions, OOB score (if applicable), and feature importances


def apply_KNN(df_test, df_train, target_label, feature_cols, no_gridsearch=False):
    """Performs GridSearchCV (or default fit) for KNN, predicts, returns results."""
    print(f"Applying K Nearest Neighbors {'(Default Params)' if no_gridsearch else '(GridSearchCV)'}...")
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

    # Check for empty data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # KNN doesn't provide feature importances directly
    oob = None # Not applicable to KNN
    best_params = None
    best_score = np.nan
    search_time = np.nan

    # --- Define Default Parameters for KNN step ---
    default_params = {
        'n_neighbors': 5,       # Sklearn default
        #'weights': 'uniform',   # Sklearn default
        #'metric': 'minkowski'   # Sklearn default (Euclidean distance with p=2)
        # Add other defaults if needed: 'algorithm': 'auto', 'leaf_size': 30, 'p': 2
    }

    # KNN requires feature scaling, so we use a Pipeline
    if no_gridsearch:
        # --- Fit with Default Parameters using Pipeline ---
        print("  Skipping GridSearchCV. Using default KNN parameters within Pipeline.")
        start_time = time.time()
        # Instantiate pipeline with StandardScaler and KNN using default_params
        pipe = Pipeline([
            ('scaler', StandardScaler()), # Scale features
            ('knn', KNeighborsRegressor(**default_params)) # Apply KNN
        ])
        pipe.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  Default KNN Pipeline fitted in {search_time:.2f} seconds.")
        y_pred = pipe.predict(X_test)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Fit Time (s)": search_time }
        # Pass the used default_params for KNN to logger
        log_experiment("K Nearest Neighbors", default_params, metrics, grid_searched=False)

    else:
        # --- Perform GridSearchCV using Pipeline ---
        # Base pipeline for GridSearchCV
        pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor())])
        # Define parameter grid (note the 'knn__' prefix for pipeline steps)
        param_grid = {
            'knn__n_neighbors': [5, 10, 15],
            'knn__weights': ['uniform', 'distance'], # Test different weighting schemes
        }
        print(f"  Parameter Grid for KNN GridSearchCV:\n  {param_grid}")
        start_time = time.time()
        # Setup GridSearchCV for the pipeline
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   cv=INNER_CV_FOLDS, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train) # Fit the entire pipeline
        search_time = time.time() - start_time
        print(f"  GridSearchCV finished in {search_time:.2f} seconds.")

        # Extract results
        best_pipe = grid_search.best_estimator_ # The best pipeline found
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        y_pred = best_pipe.predict(X_test) # Predict using the best pipeline
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Best Inner CV Score (Neg MSE)": best_score,
                    "GridSearch Time (s)": search_time }
        # Log results, pass empty dict {} for fixed params as none were explicitly set for grid search base model
        log_experiment("K Nearest Neighbors", {}, metrics, best_params=best_params, grid_searched=True)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp # oob and f_imp are None for KNN


def apply_SVM(df_test, df_train, target_label, feature_cols, no_gridsearch=False):
    """Performs GridSearchCV (or default fit) for SVR, predicts, returns results."""
    print(f"Applying Support Vector Machine (SVR) {'(Default Params)' if no_gridsearch else '(GridSearchCV)'}...")
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

    # Check for empty data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # Feature importance not standard/easy for kernel SVR
    oob = None # Not applicable
    best_params = None
    best_score = np.nan
    search_time = np.nan

    # --- Define Default Parameters for SVR step ---
    # Includes the previously fixed epsilon
    default_params = {
        'kernel': 'rbf',        # Sklearn default kernel
        'C': 1.0,               # Sklearn default regularization parameter
        'gamma': 'scale',       # Sklearn default kernel coefficient
        #'epsilon': 5e-6         # Keep fixed as per original script's apparent intention
    }

    # SVR also requires feature scaling, use a Pipeline
    if no_gridsearch:
        # --- Fit with Default Parameters using Pipeline ---
        print("  Skipping GridSearchCV. Using default SVR parameters within Pipeline.")
        start_time = time.time()
        # Instantiate pipeline with StandardScaler and SVR using default_params
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**default_params))
        ])
        pipe.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  Default SVR Pipeline fitted in {search_time:.2f} seconds.")
        y_pred = pipe.predict(X_test)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Fit Time (s)": search_time }
        # Pass the used default_params for SVR to logger
        log_experiment("Support Vector Machine (SVR)", default_params, metrics, grid_searched=False)

    else:
        # --- Perform GridSearchCV using Pipeline ---
        # Base pipeline for GridSearchCV - still fix epsilon here if desired
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR(epsilon=default_params['epsilon']))])
        # Grid search over other parameters (C, gamma, potentially kernel)
        param_grid = {
            'svr__C': [0.1, 1], # Test different regularization strengths
            'svr__kernel': ['rbf'], # Only testing rbf here, could add 'linear', 'poly' etc.
            'svr__gamma': ['scale', 0.1, 1], # Test scale vs specific values for gamma,
            'svr__epsilon': [5e-6, 0.1, 1.0]
        }
        print(f"  Parameter Grid for SVR GridSearchCV:\n  {param_grid}")
        start_time = time.time()
        # Setup GridSearchCV for the pipeline
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   cv=INNER_CV_FOLDS, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  GridSearchCV finished in {search_time:.2f} seconds.")

        # Extract results
        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        y_pred = best_pipe.predict(X_test)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Best Inner CV Score (Neg MSE)": best_score,
                    "GridSearch Time (s)": search_time }
        # Log results, pass fixed epsilon used in base estimator
        log_experiment("Support Vector Machine (SVR)", {'epsilon': default_params['epsilon']}, metrics, best_params=best_params, grid_searched=True)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp # oob and f_imp are None


def apply_BayesianRidge(df_test, df_train, target_label, feature_cols, no_gridsearch=False):
    """Performs GridSearchCV (or default fit) for BayesianRidge, predicts, returns results."""
    print(f"Applying Bayesian Ridge {'(Default Params)' if no_gridsearch else '(GridSearchCV)'}...")
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

    # Check for empty data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # Feature importances (absolute coefficient values)
    oob = None # Not applicable
    best_params = None
    best_score = np.nan
    search_time = np.nan

    # --- Define Default Parameters for BayesianRidge step ---
    # Defaults often work well, can leave empty or specify if needed
    default_params = {
         # 'alpha_1': 1e-6, # Default shape parameter for gamma distribution prior over alpha
         # 'lambda_1': 1e-6 # Default shape parameter for gamma distribution prior over lambda
         # Keep empty to use sklearn's internal defaults, which are generally robust
    }

    # Bayesian Ridge (like other linear models) benefits from scaling
    if no_gridsearch:
        # --- Fit with Default Parameters using Pipeline ---
        print("  Skipping GridSearchCV. Using default BayesianRidge parameters within Pipeline.")
        start_time = time.time()
        # Instantiate pipeline with StandardScaler and BR using default_params (which might be empty)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('br', BayesianRidge(**default_params))
        ])
        pipe.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  Default BayesianRidge Pipeline fitted in {search_time:.2f} seconds.")
        y_pred = pipe.predict(X_test)
        # Get feature importances (absolute values of coefficients)
        f_imp = np.abs(pipe.named_steps['br'].coef_)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Fit Time (s)": search_time }
        # Pass the used default_params for BR to logger
        log_experiment("Bayesian Ridge", default_params if default_params else "Sklearn Defaults", metrics, grid_searched=False)

    else:
        # --- Perform GridSearchCV using Pipeline ---
        # Base pipeline for GridSearchCV
        pipe = Pipeline([('scaler', StandardScaler()), ('br', BayesianRidge())])
        # Example Grid - Tune regularization priors slightly (often not necessary)
        param_grid = {
            'br__tol': [1e-3, 1e-2,1e-4],
            #'br__alpha_1': [1e-6, 1e-5, 1e-7], # Shape parameter for alpha prior
            #'br__lambda_1': [1e-6, 1e-5, 1e-7] # Shape parameter for lambda prior
            # Can also tune alpha_2, lambda_2 (scale parameters)
        }
        print(f"  Parameter Grid for BayesianRidge GridSearchCV:\n  {param_grid}")
        start_time = time.time()
        # Setup GridSearchCV for the pipeline
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   cv=INNER_CV_FOLDS, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  GridSearchCV finished in {search_time:.2f} seconds.")

        # Extract results
        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        y_pred = best_pipe.predict(X_test)
        # Get coefs from the best model found in the pipeline
        f_imp = np.abs(best_pipe.named_steps['br'].coef_)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Best Inner CV Score (Neg MSE)": best_score,
                    "GridSearch Time (s)": search_time }
        # Log results, pass empty dict {} for fixed params
        log_experiment("Bayesian Ridge", {}, metrics, best_params=best_params, grid_searched=True)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp # oob is None


def apply_Lasso(df_test, df_train, target_label, feature_cols, no_gridsearch=False):
    """Performs GridSearchCV (or default fit) for Lasso, predicts, returns results."""
    print(f"Applying Lasso Regression {'(Default Params)' if no_gridsearch else '(GridSearchCV)'}...")
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

    # Check for empty data
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Warning: Empty training or test set after removing NaN labels/features.")
        return None, None, None

    # Initialize results
    y_pred = None
    f_imp = None # Feature importances (absolute coefficient values)
    oob = None # Not applicable
    best_params = None
    best_score = np.nan
    search_time = np.nan

    # --- Define Default Parameters for Lasso step ---
    # Includes previously fixed random_state and increased max_iter
    default_params = {
       # 'alpha': 1.0,           # Sklearn default regularization strength
        #'max_iter': 5000,       # Increased from sklearn default (1000) for convergence
       # 'random_state': 42,     # Ensures reproducibility
       # 'tol': 1e-4             # Sklearn default tolerance for stopping criterion
    }

    # Lasso also benefits from scaling
    if no_gridsearch:
        # --- Fit with Default Parameters using Pipeline ---
        print("  Skipping GridSearchCV. Using default Lasso parameters within Pipeline.")
        start_time = time.time()
        # Instantiate pipeline with StandardScaler and Lasso using default_params
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(**default_params))
        ])
        pipe.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  Default Lasso Pipeline fitted in {search_time:.2f} seconds.")
        y_pred = pipe.predict(X_test)
        # Get feature importances (absolute values of coefficients)
        f_imp = np.abs(pipe.named_steps['lasso'].coef_)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Fit Time (s)": search_time }
        # Pass the used default_params for Lasso to logger
        log_experiment("Lasso Regression", default_params, metrics, grid_searched=False)

    else:
        # --- Perform GridSearchCV using Pipeline ---
        # Base pipeline for GridSearchCV - use fixed params here for consistency
        pipe = Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(random_state=default_params['random_state'], max_iter=default_params['max_iter']))])
        # Grid search only over alpha (the main tuning parameter for Lasso)
        param_grid = {
            'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], # Test a range of regularization strengths
            #'lasso__tol': [1000, 3000, 5000]
        }
        print(f"  Parameter Grid for Lasso GridSearchCV:\n  {param_grid}")
        start_time = time.time()
        # Setup GridSearchCV for the pipeline
        grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid,
                                   cv=INNER_CV_FOLDS, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        print(f"  GridSearchCV finished in {search_time:.2f} seconds.")

        # Extract results
        best_pipe = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        y_pred = best_pipe.predict(X_test)
        # Get coefs from the best model found
        f_imp = np.abs(best_pipe.named_steps['lasso'].coef_)
        metrics = { "Test MSE": mean_squared_error(y_test, y_pred),
                    "R^2 Score": r2_score(y_test, y_pred),
                    "Best Inner CV Score (Neg MSE)": best_score,
                    "GridSearch Time (s)": search_time }
        # Log results, pass fixed params used in base estimator
        log_experiment("Lasso Regression", {'random_state': default_params['random_state'], 'max_iter': default_params['max_iter']}, metrics, best_params=best_params, grid_searched=True)

    # 4. Prepare full prediction array
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    if y_pred is not None:
        full_y_pred[test_valid_idx] = y_pred

    return full_y_pred, oob, f_imp # oob is None


# --- Data Processing Functions ---

def fit_transformation(df, target_label, baseline_ll_col, group_id_col, trans=False):
    """Transforms target label based on baseline LL (e.g., NNI tree LL).

    Transformation: target = target / -baseline_ll
    Optional: Apply exp2(transformed_target + 1)

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_label (str): Name of the target column (e.g., 'd_ll').
        baseline_ll_col (str): Name of the column with baseline log-likelihoods (e.g., 'nni_tree_ll').
        group_id_col (str): Name of the group ID column (used for context, not calculation).
        trans (bool): If True, apply the additional exp2 transformation.

    Returns:
        pd.DataFrame: DataFrame with the target column transformed.
    """
    print("Applying target transformation...")
    if baseline_ll_col not in df.columns or target_label not in df.columns:
        print(f"Error: Required columns for transformation missing ({baseline_ll_col}, {target_label}). Skipping.")
        return df

    # Ensure columns are numeric, coercing errors to NaN
    # Use .loc to modify the DataFrame directly
    df.loc[:, target_label] = pd.to_numeric(df[target_label], errors='coerce')
    df.loc[:, baseline_ll_col] = pd.to_numeric(df[baseline_ll_col], errors='coerce')

    # --- Transformation logic ---
    # Create a mask for valid transformation conditions:
    # - target is not NaN
    # - baseline is not NaN
    # - baseline is not zero (to avoid division by zero)
    mask = pd.notna(df[target_label]) & pd.notna(df[baseline_ll_col]) & (df[baseline_ll_col] != 0)

    # Apply the division transformation where the mask is True
    df.loc[mask, target_label] = df.loc[mask, target_label] / -df.loc[mask, baseline_ll_col]
    # Set target to NaN where transformation is not possible (mask is False)
    df.loc[~mask, target_label] = np.nan

    # --- Optional exp2 transformation ---
    if trans:
        print("Applying exp2 transformation...")
        # Mask for non-NaN values in the potentially transformed target
        mask_trans = pd.notna(df[target_label])
        # Handle potential overflow/domain errors for exp2
        try:
            # Apply exp2 safely. Consider clipping input values if they can become extremely large/small.
            # Example clipping (adjust range as needed):
            # clipped_values = np.clip(df.loc[mask_trans, target_label] + 1, -50, 50) # Prevents huge exp2 results
            # df.loc[mask_trans, target_label] = np.exp2(clipped_values)

            # Apply without clipping (as in original logic)
            df.loc[mask_trans, target_label] = np.exp2(df.loc[mask_trans, target_label] + 1)

        except (OverflowError, ValueError) as e:
            print(f"Warning: Error during exp2 transformation: {e}. Setting problematic values to NaN.")
            # Identify potential problematic values (e.g., very large inputs to exp2)
            # For simplicity, set all potentially transformed values back to NaN on error
            # A more refined approach might try to identify *which* values caused the error.
            df.loc[mask_trans, target_label] = np.nan

    print("Transformation finished.")
    return df


def truncate(df, group_id_col, kfold):
    """Truncates DataFrame groups to be divisible by kfold for CV splitting.

    Handles both integer kfold values and 'LOO' (Leave-One-Out).

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_id_col (str): Column name containing dataset group IDs.
        kfold (int or str): Number of folds or 'LOO'.

    Returns:
        tuple: (truncated_df, group_ids, test_batch_size)
               - truncated_df (pd.DataFrame): DataFrame with groups removed if necessary.
               - group_ids (np.ndarray): Array of unique group IDs remaining after truncation.
               - test_batch_size (int): Number of groups per test fold (1 for LOO). Returns 0 on error.
    """
    print("Truncating datasets for K-Fold/LOO...")
    if group_id_col not in df.columns:
        print(f"Error: Group ID column '{group_id_col}' not found for truncation.")
        return df, np.array([]), 0 # Return empty array and 0 batch size on error

    # Drop rows where group ID is missing before getting unique groups
    df = df.dropna(subset=[group_id_col])
    groups_ids = df[group_id_col].unique()
    n_groups = len(groups_ids)
    print(f"Initial number of dataset groups: {n_groups}")

    if n_groups == 0:
        print("Warning: No groups found after dropping NaNs in group ID column.")
        return df, groups_ids, 0 # Return empty df and 0 batch size

    # --- Handle LOO (Leave-One-Out) Case ---
    if isinstance(kfold, str) and kfold.upper() == "LOO":
        kfold_num = n_groups # In LOO, the number of folds equals the number of groups
        if kfold_num <= 0:
             print("Warning: Cannot perform LOO with 0 groups.")
             return df.reset_index(drop=True), groups_ids, 0 # Return original df (empty) and 0 batch size
        print("Using Leave-One-Out (LOO) CV strategy.")
        test_batch_size = 1 # Each fold tests exactly one group
        print(f"Number of dataset groups (folds): {n_groups}")
        print(f"Test batch size per fold: {test_batch_size}")
        # No truncation needed for LOO, return original df (with NaNs dropped) and all group IDs
        return df.reset_index(drop=True), groups_ids, test_batch_size

    # --- Handle Integer K-Fold Case ---
    elif isinstance(kfold, int):
        kfold_num = kfold
        if kfold_num <= 0:
            print("Error: KFOLD value must be a positive integer or 'LOO'.")
            # Or handle as no CV? For now, return error state.
            return df.reset_index(drop=True), groups_ids, 0 # Indicate error / no valid CV

        # Adjust kfold if it's larger than the number of groups
        if n_groups < kfold_num:
            print(f"Warning: Number of groups ({n_groups}) is less than KFOLD ({kfold_num}). Setting KFOLD to {n_groups}.")
            kfold_num = n_groups

        if kfold_num <= 0: # Double check if n_groups became 0
             print("Warning: Cannot perform CV with 0 effective folds.")
             return df.reset_index(drop=True), groups_ids, 0

        # Calculate number of groups to remove for divisibility
        ndel = n_groups % kfold_num
        if ndel != 0:
            print(f"Removing {ndel} groups to make total ({n_groups}) divisible by KFOLD={kfold_num}.")
            # Remove groups from the end of the unique groups list
            ids_to_remove = groups_ids[-ndel:]
            # Filter the DataFrame, keeping only rows whose group ID is NOT in ids_to_remove
            df = df[~df[group_id_col].isin(ids_to_remove)].copy() # Use .copy() to avoid potential SettingWithCopyWarning later
            groups_ids = df[group_id_col].unique() # Update group IDs after removal

        # Calculate the test batch size (number of groups per fold)
        final_n_groups = len(groups_ids)
        if final_n_groups == 0 or kfold_num == 0:
            print("Warning: No groups remaining after truncation or kfold is zero.")
            test_batch_size = 0
        else:
            test_batch_size = final_n_groups // kfold_num # Integer division

        print(f"Number of dataset groups after truncation: {final_n_groups}")
        print(f"Test batch size per fold: {test_batch_size}")
        return df.reset_index(drop=True), groups_ids, test_batch_size # Return truncated df and calculated batch size
    else:
        print(f"Error: Invalid kfold type: {type(kfold)}. Must be int or 'LOO'.")
        return df, groups_ids, 0 # Indicate error state

# --- Cross-Validation / Validation Set Function ---

def evaluate_model(df, target_label, feature_cols, group_id_col, model_name, kfold_value, add_params, add_mf,
                   validation_set=False, validation_set_path=None,
                   trans=False, random_choice=False, scale_score=False,
                   no_gridsearch=False): # Added no_gridsearch parameter
    """Performs K-Fold CV or evaluates on a validation set.

    Args:
        df (pd.DataFrame): The main DataFrame (used for training in validation mode, or full data for CV).
        target_label (str): Name of the target variable column.
        feature_cols (list): List of feature column names.
        group_id_col (str): Column name for dataset grouping (for CV splits).
        model_name (str): Name of the model to use ('RFR', 'KNN', 'SVM', 'BR', 'Lasso').
        kfold_value (int or str): Number of folds or 'LOO' (used only if validation_set is False).
        validation_set (bool): If True, use validation set mode. If False, use K-Fold CV.
        validation_set_path (str): Path to the validation set CSV (required if validation_set is True).
        trans (bool): Whether target transformation was applied (passed to fit_transformation for validation set).
        random_choice (bool): Passed to ds_scores.
        scale_score (bool): Passed to ds_scores (scales rank to %).
        no_gridsearch (bool): If True, skips GridSearchCV and uses default model parameters.

    Returns:
        tuple: (results_dictionary, dataframe_with_predictions)
               - results_dictionary (dict): Aggregated scores (OOB, feature importances, ranks, correlations).
               - dataframe_with_predictions (pd.DataFrame): The original df (or validation set) with a 'pred' column added.
    """
    res_dict = {} # Dictionary to store aggregated results
    oobs, f_imps = [], [] # Lists to collect OOB scores and feature importances from folds/runs
    df_out = df.copy() # Make a copy to add predictions to, preserving the original df

    # Map model name string to the corresponding application function
    model_func_map = {
        'RFR': apply_RFR,
        'KNN': apply_KNN,
        'SVM': apply_SVM,
        'BR': apply_BayesianRidge,
        'Lasso': apply_Lasso
    }
    apply_func = model_func_map.get(model_name) # Get the function based on the model_name argument
    if not apply_func:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(model_func_map.keys())}")
    print(f"Selected model function: {apply_func.__name__}") # Show which function is being called

    if not validation_set:
        # --- K-Fold Cross Validation ---
        print(f"\nStarting {kfold_value}-Fold Cross Validation (Outer Loop)...")
        # Truncate data based on group IDs and kfold value
        # Returns: df_cv (potentially truncated df), groups_ids (ids for CV), test_batch_size
        df_cv, groups_ids, test_batch_size = truncate(df_out, group_id_col, kfold_value)

        # Check if truncation/setup was successful
        if test_batch_size <= 0 or len(groups_ids) == 0:
             print(f"Warning: Cannot perform CV. test_batch_size={test_batch_size}, num_groups={len(groups_ids)}. Check data and kfold value.")
             # Ensure 'pred' column exists even if CV fails, initialized with NaNs
             if 'pred' not in df_out.columns: df_out["pred"] = np.nan
             return {}, df_out # Return empty results and original df

        # Determine actual number of folds to iterate through
        actual_k = math.ceil(len(groups_ids) / test_batch_size) if test_batch_size > 0 else 0
        print(f"Actual number of outer folds to run: {actual_k}")

        # Initialize prediction column in the CV dataframe with NaNs
        df_cv["pred"] = np.nan
        fold_num = 0
        # Iterate through the group IDs in batches defined by test_batch_size
        for low_i_idx in range(0, len(groups_ids), test_batch_size):
            fold_num += 1
            up_i_idx = low_i_idx + test_batch_size
            test_group_ids = groups_ids[low_i_idx:up_i_idx] # IDs for the current test fold
            # IDs for the training fold are all IDs *except* the test IDs
            train_group_ids = np.setdiff1d(groups_ids, test_group_ids)

            print(f"\n--- Outer Fold {fold_num}/{actual_k} ---")
            # Print only first few test IDs if list is long for brevity
            test_ids_str = str(list(test_group_ids[:5])) + ('...' if len(test_group_ids) > 5 else '')
            print(f"  Test Group IDs: {test_ids_str}")

            # Get train/test data splits for this fold based on group IDs from the *truncated* df_cv
            df_test_fold = df_cv[df_cv[group_id_col].isin(test_group_ids)].copy()
            df_train_fold = df_cv[df_cv[group_id_col].isin(train_group_ids)].copy()

            # Skip fold if either training or test set is empty
            if df_train_fold.empty or df_test_fold.empty:
                print(f"Warning: Fold {fold_num} has empty train ({len(df_train_fold)}) or test ({len(df_test_fold)}) set. Skipping.")
                continue

            print(f"  Training set size: {len(df_train_fold)}, Test set size: {len(df_test_fold)}")

            # Call the selected model function (e.g., apply_RFR), passing the no_gridsearch flag
            y_pred_fold, oob_fold, f_imp_fold = apply_func(
                df_test_fold, df_train_fold, target_label, feature_cols, add_params, add_mf, no_gridsearch=no_gridsearch
            )

            # Collect results from the fold
            if oob_fold is not None: oobs.append(oob_fold) # Collect OOB score if available
            if f_imp_fold is not None: f_imps.append(f_imp_fold) # Collect feature importances
            if y_pred_fold is not None:
                # Ensure prediction indices align with the test fold's indices before assigning
                # Use df_test_fold.index to put predictions back into the correct rows of df_cv
                if len(y_pred_fold) == len(df_test_fold.index):
                     df_cv.loc[df_test_fold.index, "pred"] = y_pred_fold
                else:
                     print(f"Warning: Prediction length mismatch in fold {fold_num}. Cannot assign predictions.")
                     print(f"  Expected length: {len(df_test_fold.index)}, Got length: {len(y_pred_fold)}")
                     # Optionally fill with NaNs or handle error differently
                     # df_cv.loc[df_test_fold.index, "pred"] = np.nan


        print("\nCross Validation Finished.")
        df_out = df_cv # The output dataframe is the one used for CV, now containing 'pred' column

    else:
        # --- Validation Set Mode ---
        print("\nUsing Validation Set strategy...")
        if not validation_set_path or not os.path.exists(validation_set_path):
            print(f"Error: Validation set path missing or not found: {validation_set_path}")
            if 'pred' not in df_out.columns: df_out["pred"] = np.nan
            return {}, df_out # Return empty results and original training df

        df_train = df_out # Training data is the input dataframe 'df' (already copied to df_out)
        print(f"Training data size: {len(df_train)}")

        # Load the validation set
        try:
            print(f"Loading validation set from: {validation_set_path}")
            df_test = pd.read_csv(validation_set_path, dtype=types_dict) # Use predefined types
            print(f"Validation set size: {len(df_test)}")
        except Exception as e:
            print(f"Error reading validation set CSV: {e}")
            if 'pred' not in df_out.columns: df_out["pred"] = np.nan
            return {}, df_out # Return empty results and original training df

        # Check required columns in validation set
        # Need features, target, group ID (for scoring), and baseline LL (if transforming)
        required_val_cols = feature_cols + [target_label, group_id_col]
        if trans: required_val_cols.append(NNI_TREE_LL_COL) # Baseline LL needed only if transforming
        missing_val_cols = [col for col in required_val_cols if col not in df_test.columns]
        if missing_val_cols:
            print(f"Error: Validation set CSV missing required columns: {missing_val_cols}")
            # Create empty pred column in test df before returning
            if 'pred' not in df_test.columns: df_test["pred"] = np.nan
            return {}, df_test # Return empty results and the loaded (but unusable) test df

        # Apply transformation to validation set if it was applied to training set
        if trans:
            print("Applying transformation to validation set...")
            # Use a copy to avoid modifying original df_test if fit_transformation fails partially
            # Pass trans=True to apply the exp2 part as well
            df_test = fit_transformation(df_test.copy(), target_label, NNI_TREE_LL_COL, group_id_col, trans=True)
        else:
            # Ensure target label is numeric even if no transformation
            df_test.loc[:, target_label] = pd.to_numeric(df_test[target_label], errors='coerce')

        # Drop rows with NaN target label *after* potential transformation in the validation set
        initial_val_rows = len(df_test)
        df_test = df_test.dropna(subset=[target_label])
        if len(df_test) < initial_val_rows:
            print(f"Dropped {initial_val_rows - len(df_test)} rows with NaN target label from validation set.")

        # Check if either set is empty after preprocessing
        if df_train.empty or df_test.empty:
            print(f"Error: Training ({len(df_train)}) or validation ({len(df_test)}) set is empty after preprocessing.")
            # Ensure 'pred' column exists in the output dataframe (which will be df_test here)
            if 'pred' not in df_test.columns: df_test["pred"] = np.nan
            df_out = df_test # Return the (potentially empty) test dataframe
            return {}, df_out

        print(f"Final training size: {len(df_train)}, Final validation size: {len(df_test)}")

        # Call the selected model function (train on df_train, predict on df_test)
        y_pred_val, oob_val, f_imp_val = apply_func(
            df_test, df_train, target_label, feature_cols, add_params, no_gridsearch=no_gridsearch
        )

        # Collect results
        if oob_val is not None: oobs.append(oob_val) # Likely None unless RFR default was run
        if f_imp_val is not None: f_imps.append(f_imp_val)
        # Add predictions to the validation set DataFrame
        if y_pred_val is not None:
             if len(y_pred_val) == len(df_test.index):
                 df_test["pred"] = y_pred_val
             else:
                 print(f"Warning: Prediction length mismatch in validation set. Cannot assign predictions.")
                 print(f"  Expected length: {len(df_test.index)}, Got length: {len(y_pred_val)}")
                 df_test["pred"] = np.nan # Assign NaNs if length mismatch
        else:
            df_test["pred"] = np.nan # Assign NaNs if prediction failed

        df_out = df_test # Output dataframe is the test set with predictions added

    # --- Calculate final scores (based on df_out, which is either df_cv or df_test) ---
    print("\nCalculating final scores...")
    # Check if 'pred' column exists and has non-NaN values before proceeding
    if 'pred' not in df_out.columns or df_out['pred'].isnull().all():
        print("Warning: 'pred' column missing or contains only NaNs in the final dataframe. Cannot calculate scores.")
        res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan
        # Handle feature importance - average if available
        valid_f_imps = [fi for fi in f_imps if fi is not None]
        res_dict['f_importance'] = np.nanmean(np.array(valid_f_imps), axis=0) if valid_f_imps else None
        # Initialize score lists/dicts as empty
        res_dict["rank_first_pred"] = {}
        res_dict["rank_first_true"] = {}
        res_dict["spearman_corr"] = []
        return res_dict, df_out # Return empty scores and the df

    # Drop rows where either prediction or true label is NaN for scoring
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
        # Calculate ranks and correlations per group using the scored dataframe
        rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df_scored, target_label, group_id_col, random_choice, scale_score)
        # Aggregate OOB scores (if any)
        res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan # Will be NaN if no OOB scores collected
        # Average feature importances collected from folds/runs
        valid_f_imps = [fi for fi in f_imps if fi is not None]
        # Average across folds (axis=0) if importances were collected
        res_dict['f_importance'] = np.nanmean(np.array(valid_f_imps), axis=0) if valid_f_imps else None
        # Store the per-group rank dictionaries and correlation list
        res_dict["rank_first_pred"] = rank_pred_by_ds
        res_dict["rank_first_true"] = rank_test_by_ds
        res_dict["spearman_corr"] = corrs

    return res_dict, df_out # Return the dictionary of results and the DataFrame with predictions

# --- Result Formatting ---

def print_and_index_results(res_dict, features):
    """Prints aggregated results and creates a summary DataFrame.

    Args:
        res_dict (dict): Dictionary containing results from evaluate_model.
        features (list): List of feature names (used for naming importance columns).

    Returns:
        pd.DataFrame: A single-row DataFrame containing the aggregated scores.
    """
    df_agg_scores = pd.DataFrame(columns=["init"]) # Create a DataFrame to hold aggregated scores
    
    # --- Spearman Correlation ---
    spearman_corrs = res_dict.get('spearman_corr', [])
    # Filter out potential NaNs before calculating mean/median
    valid_corrs = [c for c in spearman_corrs if pd.notna(c)]
    mean_corr = np.mean(valid_corrs) if valid_corrs else np.nan
    median_corr = np.median(valid_corrs) if valid_corrs else np.nan
    df_agg_scores['corr'] = valid_corrs
    print(f"\nSpearman Correlation (across {len(valid_corrs)} groups with valid scores): Mean={mean_corr:.4f}, Median={median_corr:.4f}")

    # --- Ranks ---
    ranks_pred_raw = res_dict.get('rank_first_pred', {}).values() # Rank of best predicted in true
    ranks_true_raw = res_dict.get('rank_first_true', {}).values() # Rank of best true in predicted
    # Filter NaNs from ranks
    ranks_pred = [r for r in ranks_pred_raw if pd.notna(r)]
    ranks_true = [r for r in ranks_true_raw if pd.notna(r)]
    df_agg_scores['best_predicted_ranking'] = ranks_pred
    df_agg_scores['best_empirically_ranking'] = ranks_true
    mean_rank_pred = np.mean(ranks_pred) if ranks_pred else np.nan
    median_rank_pred = np.median(ranks_pred) if ranks_pred else np.nan
    mean_rank_true = np.mean(ranks_true) if ranks_true else np.nan
    median_rank_true = np.median(ranks_true) if ranks_true else np.nan
    """
    df_agg_scores['mean_rank_best_pred'] = mean_rank_pred
    df_agg_scores['median_rank_best_pred'] = median_rank_pred
    df_agg_scores['mean_rank_best_true'] = mean_rank_true
    df_agg_scores['median_rank_best_true'] = median_rank_true
    # Assuming scale_score=True was used, ranks are percentages
    """
    all_results = []
    
    print(f"Best Predicted Rank (%): Mean={mean_rank_pred:.2f}%, Median={median_rank_pred:.2f}% (over {len(ranks_pred)} groups)")
    print(f"Best True Rank (%): Mean={mean_rank_true:.2f}%, Median={median_rank_true:.2f}% (over {len(ranks_true)} groups)")

    # --- OOB Score ---
    oob_score = res_dict.get('oob', np.nan) # Get the averaged OOB score
    #df_agg_scores['mean_oob_score'] = oob_score
    # OOB score is often not available (e.g., default RFR runs, other models, GridSearchCV)
    print(f"Mean OOB Score: {'N/A' if pd.isna(oob_score) else f'{oob_score:.4f}'}")

    # --- Feature Importances ---
    mean_importances = res_dict.get('f_importance') # Get averaged importances
    print("\nFeature Importances (averaged over folds/runs):")
    if mean_importances is not None:
        # Ensure number of importances matches number of features used
        num_expected_features = len(features) # Use the input feature list length (FEATURE_COLS)
        if len(mean_importances) == num_expected_features:
            # Create pairs of (feature_name, importance)
            # Use FEATURE_COLS which excludes 'group_id' and matches the columns used in training
            feature_importance_pairs = sorted(zip(features, mean_importances), key=lambda x: x[1], reverse=True)
            # Add each importance to the DataFrame and print
            for feature, imp_val in feature_importance_pairs:
                 colname = "imp_" + feature # Create column name like 'imp_edge_length'
                 df_agg_scores[colname] = imp_val
                 print(f"  {feature}: {imp_val:.4f}")
        else:
            print(f"  Warning: Mismatch between number of importances ({len(mean_importances)}) and expected features ({num_expected_features}). Cannot reliably assign names.")
            # Still save raw importances if needed, with generic names
            for i, imp_val in enumerate(mean_importances):
                 colname = f"imp_raw_{i}"
                 df_agg_scores[colname] = imp_val
    else:
        print("  Not available for this model or run.")

    # Report number of groups that contributed to rank/correlation scores
    num_groups_scored = len(valid_corrs) # Use the count of valid correlations as representative
    print(f"\nNumber of groups processed for rank/correlation scores: {num_groups_scored}")
    return df_agg_scores
    
# --- New function to process kfold argument ---
def process_kfold_arg(kfold_input_string):
    """Processes the kfold command line argument string.

    Args:
        kfold_input_string (str): The value passed via the --kfold argument.

    Returns:
        int or str or None: Returns integer kfold value, 'LOO' string, or None if invalid.
    """
    if isinstance(kfold_input_string, str) and kfold_input_string.upper() == "LOO":
        return "LOO" # Return the standardized string 'LOO'
    else:
        try:
            kfold_val = int(kfold_input_string)
            # Allow kfold=1 (treat as single validation split if needed, though truncate might handle it)
            if kfold_val <= 0:
                 print("Error: KFOLD value must be positive or 'LOO'.")
                 return None # Indicate error
            return kfold_val # Return the valid integer
        except (ValueError, TypeError): # Catch non-integer strings (other than 'LOO')
             print(f"Error: Invalid KFOLD value: {kfold_input_string}. Must be a positive integer or 'LOO'.")
             return None # Indicate error

# --- Main Execution Block ---

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run ML algorithm with optional GridSearchCV on combined SPR features.')
    parser.add_argument('--input_features_dir', '-i', required=True, help='Path to the directory containing feature-enriched summary CSV files (*.spr_summary.csv). Searches recursively.')
    parser.add_argument('--output_scores_csv', '-s', required=True, help='Path to save aggregated scores CSV.')
    parser.add_argument('--output_preds_csv', '-p', required=True, help='Path to save predictions CSV (includes original data + pred column).')
    parser.add_argument('--model', '-m', type=str, default='RFR', choices=['RFR', 'KNN', 'SVM', 'BR', 'Lasso'], help='ML model to use.')
    parser.add_argument('--kfold', '-k', type=str, default=str(KFOLD), help=f'Number of folds for Outer CV or "LOO" (Leave-One-Out) (default: {KFOLD}). Inner CV for GridSearchCV is fixed at {INNER_CV_FOLDS}.')
    parser.add_argument('--transform_target', '-trans', default=False, action='store_true', help='Apply transformation to target (division by baseline LL and optional exp2). Uses nni_tree_ll as baseline.')
    parser.add_argument('--validation_set', '-val', default=False, action='store_true', help='Use validation set mode instead of K-Fold CV.')
    parser.add_argument('--validation_set_path', '-valpath', type=str, default=None, help='Path to validation set CSV (required if --validation_set is used).')
    # Optional flag to disable GridSearchCV for quick tests or using only default parameters
    parser.add_argument('--no_gridsearch', action='store_true', help='Disable GridSearchCV and use default model parameters specified within the model application functions.')
    parser.add_argument('--add_params', type=int, help='Additional parameters for learning model')
    parser.add_argument('--add_mf', type=float, help='Additional mf parameters for learning model')

    args = parser.parse_args()

    # --- Initial Setup & Logging ---
    print("--- Script Start ---")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}") # Add timestamp
    print(f"Model Selected: {args.model}")
    print(f"GridSearchCV Enabled: {not args.no_gridsearch}")
    print(f"Target Transformation Enabled: {args.transform_target}")
    print(f"Evaluation Mode: {'Validation Set' if args.validation_set else 'Cross-Validation'}")

    # Validate arguments based on mode
    if args.validation_set:
        if not args.validation_set_path:
            print("Error: --validation_set_path is required when --validation_set is used.")
            sys.exit(1)
        if not os.path.exists(args.validation_set_path):
            print(f"Error: Validation file not found: {args.validation_set_path}")
            sys.exit(1)
        print(f"Validation Set Path: {args.validation_set_path}")
        kfold_val = None # kfold value is not used in validation set mode
    else:
        # Process kfold argument for Cross-Validation mode
        kfold_val = process_kfold_arg(args.kfold) # Use the helper function
        if kfold_val is None:
            sys.exit(1) # Error message already printed in process_kfold_arg
        print(f"Outer CV Folds: {kfold_val}")

    # --- Load and Merge Data ---
    input_dir = args.input_features_dir
    print(f"\nLoading and merging feature data from: {input_dir}")
    if not os.path.isdir(input_dir):
        print(f"Error: Input features directory not found: {input_dir}")
        sys.exit(1)
    # Define search pattern for recursive search
    search_pattern = os.path.join(input_dir, '**', '*.spr_summary.csv')
    print(f"Searching pattern: {search_pattern}")
    # Use glob with recursive=True to find files in subdirectories
    all_files = glob.glob(search_pattern, recursive=True)
    if not all_files:
        print(f"Error: No '*.spr_summary.csv' files found recursively in directory: {input_dir}")
        sys.exit(1)
    print(f"Found {len(all_files)} feature files to merge.")

    # Read and concatenate files
    df_list = []
    read_errors = 0
    for f in all_files:
        try:
            # Read with specified dtypes to potentially save memory and ensure types
            df_temp = pd.read_csv(f, dtype=types_dict)
            # Basic check for essential columns before appending (optional but good practice)
            # if GROUP_ID_COL in df_temp.columns and LABEL in df_temp.columns:
            df_list.append(df_temp)
            # else:
            #    print(f"Warning: File {f} missing essential columns. Skipping.")
            #    read_errors += 1
        except Exception as e:
            print(f"Warning: Error reading file {f}: {e}. Skipping file.")
            read_errors += 1

    if not df_list:
        print("Error: No data loaded. Check input files and directory structure.")
        sys.exit(1)
    if read_errors > 0:
         print(f"Warning: Skipped {read_errors} files due to read errors or missing columns.")

    # Concatenate all loaded DataFrames
    df_learning = pd.concat(df_list, ignore_index=True)
    print(f"Merged data into DataFrame with {len(df_learning)} rows and {len(df_learning.columns)} columns.")

    # --- Preprocessing ---
    print("\nPreprocessing data...")
    # Verify all expected feature columns and essential metadata columns exist in the merged DataFrame
    # NNI_TREE_LL_COL is needed for transformation baseline
    required_cols = FEATURE_COLS + [LABEL, GROUP_ID_COL, NNI_TREE_LL_COL]
    missing_cols = [col for col in required_cols if col not in df_learning.columns]
    if missing_cols:
        print(f"Error: Merged CSV missing required columns: {missing_cols}")
        sys.exit(1)

    # Apply transformation if requested.
    # If using CV, the transformation is applied *before* splitting into folds.
    # If using validation set, it's applied here to the training data, and will be applied
    # separately to the validation data inside evaluate_model.
    if args.transform_target:
        if args.validation_set:
            print("Note: Transformation will be applied to the training data now.")
            print("      It will also be applied separately to the validation set before prediction.")
        # Apply to the main df (which is training data if validation_set=True)
        df_learning = fit_transformation(df_learning, LABEL, NNI_TREE_LL_COL, GROUP_ID_COL, trans=True)
    else:
        # Ensure target label is numeric even if not transforming (handles potential string types)
        df_learning[LABEL] = pd.to_numeric(df_learning[LABEL], errors='coerce')

    # Drop rows where the target label is NaN (essential for training/evaluation)
    # This happens *after* potential transformation which might introduce NaNs
    initial_rows = len(df_learning)
    df_learning = df_learning.dropna(subset=[LABEL])
    rows_dropped = initial_rows - len(df_learning)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with NaN target label ('{LABEL}').")

    if df_learning.empty:
        print("Error: No valid data remaining after preprocessing (dropping NaN target label). Cannot train model.")
        sys.exit(1)

    # --- Run Evaluation (CV or Validation Set) ---
    print("\nStarting model evaluation...")
    if args.no_gridsearch:
         print("*** NOTE: GridSearchCV is DISABLED via --no_gridsearch flag! Using default model parameters. ***")

    start_time = time.time()
    # Call the main evaluation function
    res_dict, df_with_preds = evaluate_model(
        df=df_learning, # The preprocessed data (training data if validation mode)
        target_label=LABEL,
        feature_cols=FEATURE_COLS, # Use the defined list of actual feature column names
        group_id_col=GROUP_ID_COL, # Actual group ID column name
        model_name=args.model,
        kfold_value=kfold_val, # Integer, 'LOO', or None (if validation mode)
        validation_set=args.validation_set,
        validation_set_path=args.validation_set_path,
        trans=args.transform_target, # Pass whether transformation was applied
        add_params=args.add_params,
        add_mf=args.add_mf,
        random_choice=False, # As per original script logic (use max prediction for rank)
        scale_score=True, # As per original script logic (ranks as %)
        no_gridsearch=args.no_gridsearch # Pass the flag to disable/enable GridSearchCV
    )
    print(f"\nEvaluation total time: {time.time() - start_time:.2f} seconds")

    # --- Save Predictions ---
    output_preds_path = args.output_preds_csv
    print(f"\nSaving DataFrame with predictions to: {output_preds_path}")
    try:
        # Ensure output directory exists before saving
        os.makedirs(os.path.dirname(output_preds_path), exist_ok=True)
        # Check if the prediction column was successfully added
        if 'pred' not in df_with_preds.columns:
            print("Warning: 'pred' column not found in the final DataFrame. Saving without predictions.")
            # Optionally add a NaN column if consistency is desired for downstream processing
            # df_with_preds['pred'] = np.nan
        # Save the DataFrame (either CV results or validation set with predictions)
        df_with_preds.to_csv(output_preds_path, index=False, float_format='%.6f') # Control float precision
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions CSV to {output_preds_path}: {e}")

    # --- Print and Save Aggregated Scores ---
    print("\n--- Aggregated Results ---")
    # Pass FEATURE_COLS for correct feature importance naming in the results summary
    df_scores = print_and_index_results(res_dict, FEATURE_COLS)

    output_scores_path = args.output_scores_csv
    print(f"\nSaving aggregated scores to: {output_scores_path}")
    try:
         # Ensure output directory exists
        os.makedirs(os.path.dirname(output_scores_path), exist_ok=True)
        # Save the single-row DataFrame with aggregated scores
        df_scores.to_csv(output_scores_path, index=False, float_format='%.6f') # Control float precision
        print("Aggregated scores saved successfully.")
    except Exception as e:
        print(f"Error saving scores CSV to {output_scores_path}: {e}")

    # --- Script End ---
    print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}") # Add end timestamp
    print("--- Script End ---")
