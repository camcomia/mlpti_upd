import argparse
import datetime
import glob
import math
from statistics import mean, median

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from defs_NNI import *

def truncate(df):
    df = df.dropna()
    groups_ids = df["group_id"].unique()

    kfold = len(groups_ids) if KFOLD=="LOO" else KFOLD
    assert len(groups_ids) >= kfold
    ndel = len(groups_ids) % kfold
    
    if ndel != 0:   # i removed datasets from the end, and not randomly. from some reason..
        for group_id in groups_ids[:-ndel-1:-1]:
            df = df[df["group_id"] != group_id]

    groups_ids = df["group_id"].unique()
    new_length = len(groups_ids)
    test_batch_size = int(new_length / kfold)

    return df.reset_index(drop=True), groups_ids, test_batch_size

def split_features_label(df, features):
    attributes_df = df[features]
    label_df = df["target"]

    x = attributes_df
    y = label_df
    
    return x, y

def apply_RFR(df_test, df_train, features):
    X_train, y_train = split_features_label(df_train, features)
    X_test, y_test = split_features_label(df_test, features)

    regressor = RandomForestRegressor(n_estimators=100, max_features=0.33,  oob_score=True, n_jobs=-1).fit(X_train, y_train) # 0.33=nfeatures/3. this is like in R (instead of default=n_features)
    y_pred = regressor.predict(X_test)
    oob = regressor.oob_score_
    f_imp = regressor.feature_importances_

    return y_pred, oob, f_imp

def ds_scores(df, random, scale_score):
    rank_pred_by_ds, rank_test_by_ds = {}, {}

    label = "target"
    sp_corrs = []
    grouped_df_by_ds = df.groupby("group_id", sort=False)
    for group_id, df_by_ds in grouped_df_by_ds:
        rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", label, random, scale_score)
        rank_test_by_ds[group_id] = score_rank(df_by_ds, label, "pred", random, scale_score)

        temp_df = df_by_ds[[label, "pred"]]
        sp_corr = temp_df.corr(method='spearman').iloc[1,0]
        if sp_corr:
            sp_corrs.append(sp_corr)
        else:
            sp_corrs.append(None)

    return rank_pred_by_ds, rank_test_by_ds, sp_corrs

def score_rank(df_by_ds, sortby, locatein, random, scale_score):
    '''
    find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
    '''

    best_pred_ix = df_by_ds[sortby].idxmax()    # changed min to max!
    if random:
        best_pred_ix = np.random.choice(df_by_ds[sortby].index, 1, replace=False)[0]
    temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()   # changed ascending to False
    best_pred_rank = min(temp_df.index[temp_df["index"] == best_pred_ix].tolist())
    best_pred_rank += 1  # convert from pythonic index to position

    if scale_score:
        best_pred_rank /= len(df_by_ds[sortby].index)   # scale the rank according to rankmax
        best_pred_rank *= 100


    return best_pred_rank

def print_and_index_results(df_datasets, res_dict, features):
    #### score 1 ####
    spearman_corrs = res_dict['spearman_corr']
    df_datasets['corr'] = spearman_corrs
    print("\nSpearman Correlation:\n" + "mean:", mean([e for e in spearman_corrs if e != None if not math.isnan(e)]), ", median:",median([e for e in spearman_corrs if e != None if not math.isnan(e)]))

    #### score 2 + 3 ####
    res_vec1 = np.asarray(list(res_dict['rank_first_pred'].values())) if type(res_dict['rank_first_pred']) is dict else res_dict['rank_first_pred']
    res_vec2 = np.asarray(list(res_dict['rank_first_true'].values()))  if type(res_dict['rank_first_true']) is dict else res_dict['rank_first_true']
    df_datasets['best_predicted_ranking'] = res_vec1
    df_datasets['best_empirically_ranking'] = res_vec2
    print("\nbest predicted rank in true:\n" + "mean:",np.mean(res_vec1), ", median:", np.median(res_vec1))
    print("\nbest true rank in pred :\n" + "mean:",np.mean(res_vec2), ", median:", np.median(res_vec2))

    mean_importances = res_dict['f_importance']   # index in first row only (score foreach run and not foreach dataset)
    for i, f in enumerate(features):
        colname = "imp_" + f
        df_datasets.loc[0, colname] = mean_importances[i]

    #### additional information ####
    df_datasets.loc[0, 'oob'] = res_dict['oob']   # index in first row only (score foreach run and not foreach dataset)
    print("oob:", res_dict['oob'])
    print("ndatasets: ", len(res_vec1))


    return df_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation on training datasets")
    parser.add_argument(
        '--training_folder', '-tf', 
        required=True,
        help="Path to the training folder containing dataset CSV files"
    )
    args = parser.parse_args()

    ##########################################################################
    # Configuration
    training_folder = args.training_folder
    ##########################################################################
    print(f"Finding all dataset.csv files in '{training_folder}'...")
    dataset_paths = glob.glob(f"{training_folder}/*/dataset.csv")
    df_all = pd.DataFrame()
    ##########################################################################
    print("Reading and assigning group IDs...")
    group_id = 1
    for path in dataset_paths:
        df = pd.read_csv(path)
        df["group_id"] = group_id
        df_all = pd.concat([df_all, df], ignore_index=True)
        group_id += 1
    print (df_all.shape)
    ##########################################################################
    print("Initializing variables...")
    df, group_ids, test_batch_size = truncate(df_all)
    res_dict = {}
    oobs, f_imps = [], []
    my_y_pred = np.full(len(df), np.nan)

    # Define selected features
    # features = ['ntaxa_S1', 'ntaxa_S2', 'bl_S1_b', 'bl_S2_b',
    #    'maxBL_S1_b', 'maxBL_S2_b', 'cumBL_S1', 'cumBL_S2', 'bl_S1_a',
    #    'bl_S2_a', 'maxBL_S1_a', 'maxBL_S2_a', 'ntaxa',
    #    'tbl', 'maxBL']
    features = ['bl_S1_b', 'bl_S2_b', 'bl_S1_a', 'bl_S2_a', 'tbl']

    ##########################################################################
    print("Performing 10-fold cross-validation...")
    round_num = 1
    for low_group in group_ids[::test_batch_size]:
        print(f"Round {round_num}...")

        # Get start and end indices for test batch
        low_idx, = np.where(group_ids == low_group)
        low_idx = int(low_idx[0])
        high_idx = low_idx + test_batch_size

        test_ids = group_ids[low_idx:high_idx]
        train_ids = np.setdiff1d(group_ids, test_ids)

        df_test = df[df["group_id"].isin(test_ids)]
        df_train = df[df["group_id"].isin(train_ids)]

        y_pred, oob, f_imp = apply_RFR(df_test, df_train, features)

        oobs.append(oob)
        f_imps.append(f_imp)
        my_y_pred[df_test.index.values] = y_pred

        round_num += 1

    ##########################################################################
    print("Generating final results...\n")
    df["pred"] = my_y_pred

    rank_pred_by_ds, rank_test_by_ds, corrs = ds_scores(df, False, True)

    # Aggregate results
    res_dict["oob"] = sum(oobs) / len(oobs)
    res_dict["f_importance"] = sum(f_imps) / len(f_imps)
    res_dict["rank_first_pred"] = rank_pred_by_ds
    res_dict["rank_first_true"] = rank_test_by_ds
    res_dict["spearman_corr"] = corrs

    # Print and store summary results
    df_datasets = pd.DataFrame(columns=["init"])
    df = print_and_index_results(df_datasets, res_dict, features)
    ##########################################################################