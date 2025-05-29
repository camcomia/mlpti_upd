# Standard libraries
import os
import sys
import argparse
import glob # For finding input files
import numpy as np
import pandas as pd
from collections import OrderedDict

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


parser = argparse.ArgumentParser(description='Run RF algorithm on combined SPR features.')
parser.add_argument('--input_features_dir', '-i', required=True, help='Path to the directory containing features csv (*.spr_summary.csv).')
parser.add_argument('--output_path', '-o', required=True, help='Filename to save merged csv files for training in.')
args = parser.parse_args()

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
    try: 
        df_list.append(pd.read_csv(f, dtype=types_dict))
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

print(f"Creating final csv output: {args.output_path}")
df_learning.to_csv(args.output_path, index=False)
