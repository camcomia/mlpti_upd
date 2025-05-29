#########################################################################
##                 Copyright (C). All Rights Reserved.                   ##
##      "Harnessing machine learning to guide                            ##
##                              phylogenetic-tree search algorithms"     ##
##                                                                       ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                       ##
## Adapted script for learning algorithm, designed to work with combined ##
## feature data generated from the NNI->SPR pipeline.                    ##
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
import glob # For finding input files

# Required ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
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
    ("group_id", "orig_ds_id")
])
FEATURE_COLS = [col for key, col in FEATURES.items() if key != "group_id"]
KFOLD = 10
GROUP_ID_COL = FEATURES.get("group_id", "orig_ds_id")
ORIG_DS_LL_COL = "orig_ds_ll"
NNI_TREE_LL_COL = "nni_tree_ll"

list_str = ['prune_name', 'rgft_name', GROUP_ID_COL]
list_int = []
list_float = ['ll', NNI_TREE_LL_COL, ORIG_DS_LL_COL, LABEL] + FEATURE_COLS
types_dict = {}
for e in list_str: types_dict[e] = np.object_
for e in list_int: types_dict[e] = np.int32
for e in list_float: types_dict[e] = np.float32


# --- Helper Functions ---

def log_experiment(model_name, params, metrics):
    """Logs experiment parameters and metrics."""
    print(f"--- {model_name} Experiment ---")
    print(f"  Parameters: {params}")
    printable_metrics = {k: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for k, v in metrics.items() if v is not None}
    print(f"  Metrics: {printable_metrics}")
    print("-" * (len(model_name) + 20))

def score_rank(df_by_ds, sortby, locatein, random_choice, scale_score):
    """Finds rank of the best predicted item within the true ranking."""
    if df_by_ds.empty or sortby not in df_by_ds or locatein not in df_by_ds: return np.nan
    df_by_ds.loc[:, sortby] = pd.to_numeric(df_by_ds[sortby], errors='coerce')
    df_by_ds.loc[:, locatein] = pd.to_numeric(df_by_ds[locatein], errors='coerce')
    df_by_ds = df_by_ds.dropna(subset=[sortby, locatein])
    if df_by_ds.empty: return np.nan
    if random_choice: best_pred_ix = np.random.choice(df_by_ds.index, 1, replace=False)[0]
    else:
        try: best_pred_ix = df_by_ds[sortby].idxmax()
        except ValueError: return np.nan
    temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()
    rank_list = temp_df.index[temp_df["index"] == best_pred_ix].tolist()
    if not rank_list: return np.nan
    best_pred_rank = min(rank_list) + 1
    if scale_score: best_pred_rank = (best_pred_rank / len(temp_df)) * 100 if len(temp_df) > 0 else 0
    return best_pred_rank

def ds_scores(df, target_label, group_id_col, random_choice, scale_score):
    """Calculates ranks and correlations per dataset group."""
    rank_pred_by_ds, rank_test_by_ds, sp_corrs = {}, {}, [] # sp_corrs is a list
    if group_id_col not in df.columns or 'pred' not in df.columns: return rank_pred_by_ds, rank_test_by_ds, sp_corrs
    grouped_df_by_ds = df.groupby(group_id_col, sort=False)
    for group_id, df_by_ds_orig in grouped_df_by_ds:
        df_by_ds = df_by_ds_orig.copy()
        rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", target_label, random_choice, scale_score)
        rank_test_by_ds[group_id] = score_rank(df_by_ds, target_label, "pred", random_choice, scale_score)
        temp_df = df_by_ds[[target_label, "pred"]].dropna()
        if len(temp_df) > 1 and temp_df[target_label].nunique() > 1 and temp_df["pred"].nunique() > 1:
            sp_corrs.append(temp_df.corr(method='spearman').iloc[0, 1]) # Append float/NaN
        else: sp_corrs.append(np.nan) # Append NaN
    return rank_pred_by_ds, rank_test_by_ds, sp_corrs # Return list

def split_features_label(df, target_label, feature_cols):
    """Splits DataFrame into features (X) and labels (y)."""
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features: 
        print(f"Error: Missing features: {missing_features}")
        return None, None
    if target_label not in df.columns: 
        print(f"Error: Target '{target_label}' not found.")
        return None, None
    attributes_df = df[feature_cols]
    label_df = df[target_label]
    x = np.array(attributes_df)
    y = np.array(label_df).ravel()
    return x, y

def impute_nans_with_train_mean(X_train, X_test):
     """Imputes NaNs using column means from training data."""
     imputed_train = False
     col_mean_train = np.nanmean(X_train, axis=0)
     if np.isnan(X_train).any():
        imputed_train = True
        print("Warning: NaNs found in training features. Imputing with column mean.")
        inds = np.where(np.isnan(X_train))
        if np.isnan(col_mean_train[inds[1]]).any(): 
            print("Error: Cannot impute NaNs in training set - mean is NaN.")
            return None, None
        X_train[inds] = np.take(col_mean_train, inds[1])
     if np.isnan(X_test).any():
        print("Warning: NaNs found in test features. Imputing with training column mean.")
        col_mean_impute = np.nanmean(X_train, axis=0) if not imputed_train else col_mean_train
        inds = np.where(np.isnan(X_test))
        valid_means = np.take(col_mean_impute, inds[1])
        if np.isnan(valid_means).any(): 
            print("Error: Cannot impute NaNs in test set - training mean is NaN.")
            return None, None
        X_test[inds] = valid_means
     return X_train, X_test

def apply_RFR(df_test, df_train, target_label, feature_cols):
    """Trains RF, predicts, returns predictions and metrics."""
    X_train, y_train = split_features_label(df_train, target_label, feature_cols)
    X_test, y_test = split_features_label(df_test, target_label, feature_cols)
    if X_train is None or X_test is None: return None, None, None
    train_valid_idx = ~np.isnan(y_train)
    test_valid_idx = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid_idx], y_train[train_valid_idx]
    X_test, y_test = X_test[test_valid_idx], y_test[test_valid_idx]
    X_train, X_test = impute_nans_with_train_mean(X_train, X_test)
    if X_train is None: 
        return None, None, None
    if X_train.shape[0] == 0 or X_test.shape[0] == 0: 
        return None, None, None
    params = { "n_estimators": 70, "max_features": 0.33, "oob_score": True if X_train.shape[0] > 1 else False, "n_jobs": -1, "random_state": 42 }
    start_time = time.time()
    regressor = RandomForestRegressor(**params).fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    training_time = time.time() - start_time
    full_y_pred = np.full(len(test_valid_idx), np.nan)
    full_y_pred[test_valid_idx] = y_pred
    oob = regressor.oob_score_ if params["oob_score"] else None
    f_imp = regressor.feature_importances_
    metrics = { "Train MSE": mean_squared_error(y_train, regressor.predict(X_train)), "Test MSE": mean_squared_error(y_test, y_pred), "R^2 Score": r2_score(y_test, y_pred), "OOB Score": oob, "Training Time (s)": training_time }
    log_experiment("Random Forest", params, metrics)
    return full_y_pred, oob, f_imp

def fit_transformation(df, target_label, baseline_ll_col, group_id_col, trans=False):
    """Transforms target label based on baseline LL."""
    print("Applying target transformation...")
    if baseline_ll_col not in df.columns or target_label not in df.columns: return df
    df.loc[:, target_label] = pd.to_numeric(df[target_label], errors='coerce')
    df.loc[:, baseline_ll_col] = pd.to_numeric(df[baseline_ll_col], errors='coerce')
    mask = pd.notna(df[target_label]) & pd.notna(df[baseline_ll_col]) & (df[baseline_ll_col] != 0)
    df.loc[mask, target_label] = df.loc[mask, target_label] / -df.loc[mask, baseline_ll_col]
    df.loc[~mask, target_label] = np.nan
    if trans:
        print("Applying exp2 transformation...")
        mask_trans = pd.notna(df[target_label])
        df.loc[mask_trans, target_label] = np.exp2(df.loc[mask_trans, target_label] + 1)
    print("Transformation finished.")
    return df

def truncate(df, group_id_col, kfold):
    """Truncates DataFrame groups to be divisible by kfold."""
    print("Truncating datasets for K-Fold...")
    df = df.dropna(subset=[group_id_col])
    groups_ids = df[group_id_col].unique()
    n_groups = len(groups_ids)
    print(f"Initial number of dataset groups: {n_groups}")
    if kfold == "LOO": kfold = n_groups
    elif not isinstance(kfold, int): return df, groups_ids, 0
    if n_groups == 0: return df, groups_ids, 0
    if isinstance(kfold, int) and kfold > 1 and n_groups < kfold: kfold = n_groups
    if not isinstance(kfold, int) or kfold <= 0:
         test_batch_size = 1 if kfold=="LOO" else 0
         return df.reset_index(drop=True), groups_ids, test_batch_size
    ndel = n_groups % kfold
    if ndel != 0:
        print(f"Removing {ndel} groups to make total divisible by KFOLD={kfold}.")
        ids_to_remove = groups_ids[-ndel:]
        df = df[~df[group_id_col].isin(ids_to_remove)].copy()
        groups_ids = df[group_id_col].unique()
    final_n_groups = len(groups_ids)
    test_batch_size = final_n_groups // kfold if kfold > 0 else 0
    print(f"Number of dataset groups after truncation: {final_n_groups}")
    print(f"Test batch size per fold: {test_batch_size}")
    return df.reset_index(drop=True), groups_ids, test_batch_size

def evaluate_model(df, target_label, feature_cols, group_id_col, model_name, kfold_value,
                   validation_set=False, validation_set_path=None,
                   trans=False, random_choice=False, scale_score=False):
    """Performs K-Fold CV or evaluates on a validation set."""
    res_dict = {}
    oobs, f_imps = [], []
    df_out = df.copy()

    if not validation_set:
        # --- K-Fold CV ---
        print(f"Starting {kfold_value}-Fold Cross Validation...")
        df_cv, groups_ids, test_batch_size = truncate(df_out, group_id_col, kfold_value)
        actual_k = len(groups_ids) if kfold_value == "LOO" else kfold_value
        if (test_batch_size == 0 and actual_k > 0) or len(groups_ids) == 0:
            print(f"Warning: Cannot perform CV.")
            df_out["pred"] = np.nan
            return {}, df_out
        if kfold_value == "LOO": test_batch_size = 1
        df_cv["pred"] = np.nan
        fold_num = 0
        for low_i_idx in range(0, len(groups_ids), test_batch_size):
            fold_num += 1
            up_i_idx = low_i_idx + test_batch_size
            test_group_ids = groups_ids[low_i_idx:up_i_idx]
            train_group_ids = np.setdiff1d(groups_ids, test_group_ids)
            df_test_fold = df_cv[df_cv[group_id_col].isin(test_group_ids)].copy()
            df_train_fold = df_cv[df_cv[group_id_col].isin(train_group_ids)].copy()
            if df_train_fold.empty or df_test_fold.empty: continue

            if model_name == 'RFR': 
                y_pred_fold, oob_fold, f_imp_fold = apply_RFR(df_test_fold, df_train_fold, target_label, feature_cols)
            else: 
                print(f"Warning: Model '{model_name}' not supported. Skipping fold.")
                continue

            if oob_fold is not None: 
                oobs.append(oob_fold)
            if f_imp_fold is not None: 
                f_imps.append(f_imp_fold)
            if y_pred_fold is not None: 
                df_cv.loc[df_test_fold.index, "pred"] = y_pred_fold
        print("Cross Validation Finished.")
        df_out = df_cv
    else:
        # --- Validation Set ---
        print("Using Validation Set strategy...")
        if not validation_set_path or not os.path.exists(validation_set_path): 
            print(f"Error: Validation set path missing or not found: {validation_set_path}")
            return {}, df_out
        df_train = df_out
        print(f"Training data size: {len(df_train)}")
        try:
            print(f"Loading validation set from: {validation_set_path}")
            df_test = pd.read_csv(validation_set_path, dtype=types_dict)
            print(f"Validation set size: {len(df_test)}")
        except Exception as e: 
            print(f"Error reading validation set CSV: {e}")
            return {}, df_out
        required_val_cols = feature_cols + [target_label, group_id_col, NNI_TREE_LL_COL]
        missing_val_cols = [col for col in required_val_cols if col not in df_test.columns]
        if missing_val_cols: 
            print(f"Error: Validation set CSV missing required columns: {missing_val_cols}")
            return {}, df_out
        if trans: 
            df_test = fit_transformation(df_test.copy(), target_label, NNI_TREE_LL_COL, group_id_col, trans=True)
        else: 
            df_test[target_label] = pd.to_numeric(df_test[target_label], errors='coerce')
            
        df_test = df_test.dropna(subset=[target_label])
        if df_test.empty: 
            print("Error: No valid data remaining in validation set.")
            return {}, df_out

        if model_name == 'RFR': 
            y_pred_val, oob_val, f_imp_val = apply_RFR(df_test, df_train, target_label, feature_cols)
        else: 
            print(f"Warning: Model '{model_name}' not supported.")
            y_pred_val, oob_val, f_imp_val = None, None, None

        if oob_val is not None: oobs.append(oob_val)
        if f_imp_val is not None: f_imps.append(f_imp_val)
        if y_pred_val is not None: df_test["pred"] = y_pred_val
        else: df_test["pred"] = np.nan
        df_out = df_test

    # --- Calculate final scores ---
    print("Calculating final scores...")
    df_scored = df_out.dropna(subset=['pred', target_label])
    rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df_scored, target_label, group_id_col, random_choice, scale_score)
    res_dict['oob'] = np.nanmean(oobs) if oobs else np.nan
    res_dict['f_importance'] = np.nanmean(np.array(f_imps), axis=0) if f_imps and any(fi is not None for fi in f_imps) else None
    res_dict["rank_first_pred"] = rank_pred_by_ds
    res_dict["rank_first_true"] = rank_test_by_ds
    res_dict["spearman_corr"] = corrs
    return res_dict, df_out

def print_and_index_results(res_dict, features):
    """Prints aggregated results and creates a summary DataFrame."""
    df_agg_scores = pd.DataFrame(index=[0])

    #### score 1: Spearman Correlation ####
    spearman_corrs = res_dict.get('spearman_corr', []) # Default to empty list
    mean_corr = np.nanmean(spearman_corrs) if spearman_corrs else np.nan
    median_corr = np.nanmedian(spearman_corrs) if spearman_corrs else np.nan
    df_agg_scores['mean_spearman_corr'] = mean_corr
    df_agg_scores['median_spearman_corr'] = median_corr
    print(f"\nSpearman Correlation (across groups): Mean={mean_corr:.4f}, Median={median_corr:.4f}")

    #### score 2 + 3: Ranks ####
    ranks_pred = list(res_dict.get('rank_first_pred', {}).values())
    ranks_true = list(res_dict.get('rank_first_true', {}).values())
    mean_rank_pred = np.nanmean(ranks_pred) if ranks_pred else np.nan
    median_rank_pred = np.nanmedian(ranks_pred) if ranks_pred else np.nan
    mean_rank_true = np.nanmean(ranks_true) if ranks_true else np.nan
    median_rank_true = np.nanmedian(ranks_true) if ranks_true else np.nan
    df_agg_scores['mean_rank_best_pred'] = mean_rank_pred
    df_agg_scores['median_rank_best_pred'] = median_rank_pred
    df_agg_scores['mean_rank_best_true'] = mean_rank_true
    df_agg_scores['median_rank_best_true'] = median_rank_true
    print(f"Best Predicted Rank (in True): Mean={mean_rank_pred:.2f}%, Median={median_rank_pred:.2f}%")
    print(f"Best True Rank (in Predicted): Mean={mean_rank_true:.2f}%, Median={median_rank_true:.2f}%")

    #### OOB Score ####
    oob_score = res_dict.get('oob', np.nan)
    df_agg_scores['mean_oob_score'] = oob_score
    print(f"Mean OOB Score: {oob_score:.4f}")

    #### Feature Importances ####
    mean_importances = res_dict.get('f_importance')
    print("\nFeature Importances:")
    if mean_importances is not None and len(mean_importances) == len(features):
        sorted_indices = np.argsort(mean_importances)[::-1]
        for i in sorted_indices:
            if i < len(features): 
                colname = "imp_" + features[i]
                imp_val = mean_importances[i]
                df_agg_scores[colname] = imp_val
                print(f"  {features[i]}: {imp_val:.4f}")
            else: 
                print(f"Warning: Importance index {i} out of bounds.")
    else: print("  Not available or feature mismatch.")
    print(f"\nNumber of groups processed for scores: {len(ranks_pred)}")
    return df_agg_scores

def process_kfold_arg(kfold_input_string):
    """Processes the kfold command line argument string."""
    if kfold_input_string.upper() == "LOO": return "LOO"
    else:
        try:
            kfold_val = int(kfold_input_string)
            if kfold_val <= 0: 
                print("Error: KFOLD value must be positive or 'LOO'.")
                return None
            return kfold_val
        except ValueError: 
            print(f"Error: Invalid KFOLD value: {kfold_input_string}. Must be integer or 'LOO'.")
            return None

# --- Main Execution Block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RF algorithm on combined SPR features.')
    parser.add_argument('--input_features_dir', '-i', required=True, help='Path to the directory containing feature-enriched summary CSV files (*.spr_summary.csv).')
    parser.add_argument('--output_scores_csv', '-s', required=True, help='Path to save aggregated scores CSV.')
    parser.add_argument('--output_preds_csv', '-p', required=True, help='Path to save predictions CSV.')
    parser.add_argument('--model', '-m', type=str, default='RFR', choices=['RFR'], help='ML model to use (currently only RFR supported).')
    parser.add_argument('--kfold', '-k', type=str, default=str(KFOLD), help=f'Number of folds for CV or "LOO" (default: {KFOLD}).')
    parser.add_argument('--transform_target', '-trans', default=False, action='store_true', help='Apply transformation to target.')
    parser.add_argument('--validation_set', '-val', default=False, action='store_true', help='Use validation set instead of CV.')
    parser.add_argument('--validation_set_path', '-valpath', type=str, default=None, help='Path to validation set CSV.')
    args = parser.parse_args()

    if args.validation_set and not args.validation_set_path: 
        print("Error: --validation_set_path required with --validation_set.")
        sys.exit(1)
    if args.validation_set and not os.path.exists(args.validation_set_path): 
        print(f"Error: Validation file not found: {args.validation_set_path}")
        sys.exit(1)

    kfold_val = None
    if not args.validation_set:
        kfold_val = process_kfold_arg(args.kfold)
        if kfold_val is None: sys.exit(1)

    # --- Load and Merge Data ---
    input_dir = args.input_features_dir
    print(f"Loading and merging feature data from: {input_dir}")
    if not os.path.isdir(input_dir): 
        print(f"Error: Input features directory not found: {input_dir}")
        sys.exit(1)
    search_pattern = os.path.join(input_dir, '**', '*.spr_summary.csv')
    print(f"Searching pattern: {search_pattern}")
    all_files = glob.glob(search_pattern, recursive=True)
    if not all_files: 
        print(f"Error: No '*.spr_summary.csv' files found recursively in directory: {input_dir}")
        sys.exit(1)
    print(f"Found {len(all_files)} feature files to merge.")
    df_list = []
    for f in all_files:
        try: df_list.append(pd.read_csv(f, dtype=types_dict))
        except Exception as e: 
            print(f"Error reading file {f}: {e}")
            sys.exit(1)
    if not df_list: 
        print("Error: No data loaded after attempting to read files.")
        sys.exit(1)
    df_learning = pd.concat(df_list, ignore_index=True)
    print(f"Merged data into DataFrame with {len(df_learning)} rows.")

    # --- Preprocessing ---
    required_cols = FEATURE_COLS + [LABEL, GROUP_ID_COL, NNI_TREE_LL_COL]
    missing_cols = [col for col in required_cols if col not in df_learning.columns]
    if missing_cols: 
        print(f"Error: Merged CSV missing required columns: {missing_cols}")
        sys.exit(1)

    if args.transform_target:
         df_learning = fit_transformation(df_learning, LABEL, NNI_TREE_LL_COL, GROUP_ID_COL, trans=True)
    else: 
        df_learning[LABEL] = pd.to_numeric(df_learning[LABEL], errors='coerce')

    initial_rows = len(df_learning)
    df_learning = df_learning.dropna(subset=[LABEL])
    if len(df_learning) < initial_rows: 
        print(f"Dropped {initial_rows - len(df_learning)} rows with NaN target label.")
    if df_learning.empty: 
        print("Error: No valid data remaining after preprocessing.")
        sys.exit(1)

    # --- Run Evaluation (CV or Validation Set) ---
    start_time = time.time()
    res_dict, df_with_preds = evaluate_model(
        df=df_learning, target_label=LABEL, feature_cols=FEATURE_COLS,
        group_id_col=GROUP_ID_COL, model_name=args.model, kfold_value=kfold_val,
        validation_set=args.validation_set, validation_set_path=args.validation_set_path,
        trans=args.transform_target, random_choice=False, scale_score=True
    )
    print(f"Evaluation total time: {time.time() - start_time:.2f} seconds")

    # --- Save Predictions ---
    print(f"Saving DataFrame with predictions to: {args.output_preds_csv}")
    try:
        if 'pred' not in df_with_preds.columns: print("Warning: 'pred' column not found.")
        df_with_preds.to_csv(args.output_preds_csv, index=False)
    except Exception as e: 
        print(f"Error saving predictions CSV: {e}")

    # --- Print and Save Aggregated Scores ---
    print("\n--- Aggregated Results ---")
    df_scores = print_and_index_results(res_dict, FEATURE_COLS)
    print(f"Saving aggregated scores to: {args.output_scores_csv}")
    try:
        df_scores.to_csv(args.output_scores_csv, index=False)
    except Exception as e: print(f"Error saving scores CSV: {e}")

    print("\nLearning process finished.")

