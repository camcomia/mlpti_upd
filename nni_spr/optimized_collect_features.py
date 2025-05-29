#########################################################################
##                 Copyright (C). All Rights Reserved.                   ##
##      "Harnessing machine learning to guide                            ##
##                              phylogenetic-tree search algorithms"     ##
##                                                                       ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                       ##
## Adapted script to collect features based on outputs from              ##
## run_spr_on_tree.py (processing results for one NNI tree at a time).   ##
## Modifies the input summary CSV file by adding feature columns.        ##
#########################################################################

# Standard libraries
import os
import sys # For exit()
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
import random # For potential temporary naming if needed
from concurrent.futures import ProcessPoolExecutor

# Required external libraries
try:
    from ete3 import Tree
except ImportError:
    print("Error: ete3 library not found. Please install it (`pip install ete3`).")
    sys.exit(1)

# --- Constants (Define locally or import from defs_PhyAI) ---
LABEL = "d_ll" # Base label for delta log-likelihood (NOW relative to orig_ds_ll)
FEATURES = OrderedDict([
    ("bl", "edge_length"), ("longest", "longest_branch"), ("tbl_p","tbl_pruned"),("tbl_r","tbl_remaining"),
    ("longest_p","longest_pruned"),("longest_r","longest_remaining"), ("top_dist","topology_dist_between"), ("bl_dist","tbl_dist_between"),
    ("res_bl", "res_tree_edge_length"), ("ntaxa_p", "num_taxa_prune"), ("ntaxa_r", "num_taxa_rgft"),
    ("group_id", "orig_ds_id") # Need review if orig_ds_id or should this be the nni path instead, currently using nni tree
])
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"
FEATURES_RGFT_ONLY = ["top_dist", "bl_dist", "res_bl"] # Features added only for rgft context in original

# Define names for features derived from the regraft site calculation
RGFT_SITE_SUFFIX = "_rgftSite"
RGFT_FEATURES = OrderedDict([
    ("bl", "edge_length" + RGFT_SITE_SUFFIX),
    ("ntaxa_p", "num_taxa_prune" + RGFT_SITE_SUFFIX),
    ("ntaxa_r", "num_taxa_rgft" + RGFT_SITE_SUFFIX),
    ("tbl_p", "tbl_pruned" + RGFT_SITE_SUFFIX),
    ("tbl_r", "tbl_remaining" + RGFT_SITE_SUFFIX),
    ("longest_p", "longest_pruned" + RGFT_SITE_SUFFIX),
    ("longest_r", "longest_remaining" + RGFT_SITE_SUFFIX) # Approx
])
# Define column names for baseline LLs expected from run_spr_on_tree V10+
NNI_TREE_LL_COL = "nni_tree_ll"
ORIG_DS_LL_COL = "orig_ds_ll"
# Define column name for the group ID
GROUP_ID_COL = FEATURES.get("group_id", "orig_ds_id") # Get from FEATURES or default


# --- Feature Extraction Helper Functions (Mostly from original) ---

def get_newick_tree(tree_path):
    """Reads a Newick tree string from a file."""
    if not os.path.exists(tree_path):
        print(f"Error: Tree file not found at {tree_path}")
        return None
    try:
        with open(tree_path, 'r') as tree_fpr:
            tree_str = tree_fpr.read().strip()
        # Basic validation: Check for parentheses
        if not tree_str or '(' not in tree_str or ')' not in tree_str:
             # print(f"Warning: Content of {tree_path} doesn't look like a valid Newick string: {tree_str[:100]}")
             pass # Allow potentially simple strings through
        return tree_str
    except Exception as e:
        print(f"Error reading tree file {tree_path}: {e}")
        return None

def get_branch_lengths(tree):
    """Gets list of branch lengths from an ETE tree object."""
    if not isinstance(tree, Tree):
        # print("Error: get_branch_lengths expects an ETE Tree object.")
        return []
    branches = []
    for node in tree.traverse():
        # Include branch length if node has a parent (standard edges)
        if node.up:
            branches.append(node.dist)
    return branches

def get_total_branch_lengths(tree):
    """Calculates total branch length of an ETE tree object."""
    branches = get_branch_lengths(tree)
    return sum(branches)

def dist_between_nodes(t, node1):
    """Calculates topological and branch length distances from node1 to others."""
    nleaves_between, tbl_between = {},{}
    node1_name = node1.name
    if not node1_name:
        # This shouldn't happen if names are assigned, but good to check
        # print("Warning: Node1 lacks a name for distance calculation.")
        return nleaves_between, tbl_between

    for node2 in t.traverse():
        # Skip comparison with self or the root node
        if node2 == node1 or node2.is_root(): continue
        nname2 = node2.name
        if not nname2: continue # Skip target nodes without names

        try:
            # Calculate distances; +1 for topology edges
            nleaves_between[nname2] = node1.get_distance(node2, topology_only=True) + 1
            tbl_between[nname2] = node1.get_distance(node2, topology_only=False)
        except Exception as e:
            # Errors can occur if nodes are in disconnected parts after certain operations
            # print(f"Warning: Could not get distance between {node1_name} and {nname2}: {e}")
            pass # Assign nothing if distance cannot be calculated

    return nleaves_between, tbl_between

def init_recursive_features(t):
    """Initializes cumulative features (BL, maxBL, ntaxa) on tree nodes."""
    if not isinstance(t, Tree):
         # print("Error: init_recursive_features expects an ETE Tree object.")
         return
    if not t or t.is_leaf():
        if t:
             t.add_feature("cumBL", 0)
             t.add_feature("maxBL", 0)
             t.add_feature("ntaxa", 1)
        return

    for node in t.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("cumBL", 0)
            node.add_feature("maxBL", 0)
            node.add_feature("ntaxa", 1)
        else:
            # Calculate based on available children with features
            child_cumbl = sum(getattr(c, "cumBL", 0) + c.dist for c in node.children)
            child_maxbl_list = [getattr(c, "maxBL", 0) for c in node.children] + [c.dist for c in node.children]
            child_maxbl = max(child_maxbl_list) if child_maxbl_list else 0
            child_ntaxa = sum(getattr(c, "ntaxa", 0) for c in node.children)

            node.add_feature("cumBL", child_cumbl)
            node.add_feature("maxBL", child_maxbl)
            node.add_feature("ntaxa", child_ntaxa)


def calc_leaves_features(tree_input, calc_type):
    """
    Calculates features for all potential prune/regraft operations on a given tree.
    """
    if calc_type != 'prune' and calc_type != 'rgft':
        # Request for specific branch length (res_bl calculation)
        try:
            # Input might be string or already Tree object
            if isinstance(tree_input, str):
                # Check if string is empty or invalid before parsing
                if not tree_input or '(' not in tree_input:
                     # print(f"Warning: Invalid Newick string provided for res_bl calc: {tree_input[:100]}")
                     return np.nan
                t_neighbor = Tree(tree_input, format=1)
            elif isinstance(tree_input, Tree):
                t_neighbor = tree_input
            else:
                 # print(f"Error: Invalid input type for res_bl calc: {type(tree_input)}")
                 return np.nan

            target_node = t_neighbor.search_nodes(name=calc_type)
            if target_node:
                # Return distance to parent (branch length above the node)
                # Ensure node has parent before accessing dist
                return target_node[0].dist if target_node[0].up else 0.0
            else:
                # print(f"Warning: Node '{calc_type}' not found in neighbor tree for res_bl calc.")
                return np.nan # Return NaN for missing data
        except Exception as e:
            # Catch potential ETE parsing errors or other issues
            # print(f"Error calculating res_bl for node '{calc_type}': {e}")
            return np.nan

    # --- Common Setup for 'prune' and 'rgft' ---
    # Get Newick string if path provided
    tree_str = tree_input if isinstance(tree_input, str) and "(" in tree_input else get_newick_tree(tree_input)
    if not tree_str: return None

    try:
        t = Tree(newick=tree_str, format=1)
        # Ensure internal nodes have names for lookup later
        i = 0
        for node in t.traverse():
            if not node.is_leaf() and not node.name:
                node.name = f"TempFeatNode_{i}"
                i += 1
    except Exception as e:
        return None

    ntaxa_total = len(t) # Count leaves
    tbl_total = get_total_branch_lengths(t)

    # Initialize dictionaries
    name2bl, name2ntaxa_p, name2ntaxa_r = {}, {}, {}
    name2tbl_p, name2tbl_r = {}, {}
    name2longest_p, name2longest_r = {}, {}
    names2topology_dist, names2bl_dist = {}, {}

    init_recursive_features(t) # Calculate base features on all nodes

    potential_sites = [node for node in t.traverse() if node.up] # Consider all nodes with parents
    if not potential_sites:
         # print("Warning: Tree has no non-root nodes for feature calculation.")
         return None

    all_bls = [n.dist for n in potential_sites] # Includes leaves here
    overall_max_bl = max(all_bls) if all_bls else 0

    for node in potential_sites:
        nname = node.name
        if not nname: continue # Skip unnamed nodes

        bl = node.dist
        name2bl[nname] = bl

        # Use features calculated by init_recursive_features
        ntaxa_p = getattr(node, "ntaxa", 0) # Default to 0 if feature missing
        tbl_p = getattr(node, "cumBL", 0) + bl # Default cumBL to 0
        longest_p = max(getattr(node, "maxBL", 0), bl) # Default maxBL to 0

        name2ntaxa_p[nname] = ntaxa_p
        name2tbl_p[nname] = tbl_p
        name2longest_p[nname] = longest_p

        ntaxa_r = ntaxa_total - ntaxa_p
        tbl_r = tbl_total - tbl_p
        longest_r = overall_max_bl # Approximation

        name2ntaxa_r[nname] = ntaxa_r
        name2tbl_r[nname] = tbl_r
        name2longest_r[nname] = longest_r

        if calc_type == "prune":
            # This node ('node') is the potential prune site
            node_top_dist, node_bl_dist = dist_between_nodes(t, node)
            names2topology_dist[nname] = node_top_dist
            names2bl_dist[nname] = node_bl_dist

    # --- Compile results into dictionary ---
    features_dict = OrderedDict()
    features_dict[FEATURES["bl"]] = name2bl
    features_dict[FEATURES["longest"]] = overall_max_bl
    features_dict[FEATURES["ntaxa_p"]] = name2ntaxa_p
    features_dict[FEATURES["ntaxa_r"]] = name2ntaxa_r
    features_dict[FEATURES["tbl_p"]] = name2tbl_p
    features_dict[FEATURES["tbl_r"]] = name2tbl_r
    features_dict[FEATURES["longest_p"]] = name2longest_p
    features_dict[FEATURES["longest_r"]] = name2longest_r
    if calc_type == "prune":
        features_dict[FEATURES["top_dist"]] = names2topology_dist
        features_dict[FEATURES["bl_dist"]] = names2bl_dist
    return features_dict


# --- Main Processing Logic ---

def process_spr_results(nni_tree_path, summary_path, newick_path, orig_ds_id):
    """
    Reads SPR summary results, calculates features relative to the NNI tree,
    adds features and orig_ds_id as new columns to the summary DataFrame,
    and saves it back to the original summary file path.
    """
    # print(f"Processing features for NNI tree: {nni_tree_path}")
    print(f"Reading summary: {summary_path}")
    print(f"Reading newicks: {newick_path}")

    # --- Read Input Data ---
    try:
        df_summary = pd.read_csv(summary_path)
        required_summary_cols = ['prune_name', 'rgft_name', 'll', NNI_TREE_LL_COL, ORIG_DS_LL_COL]
        if not all(col in df_summary.columns for col in required_summary_cols):
             print(f"Error: Summary file {summary_path} missing required columns.")
             return False
    except Exception as e:
        print(f"Error reading summary file {summary_path}: {e}")
        return False

    try:
        df_newick = pd.read_csv(newick_path)
        required_newick_cols = ['index', 'prune_name', 'rgft_name', 'newick']
        if not all(col in df_newick.columns for col in required_newick_cols):
             print(f"Error: Newick file {newick_path} missing required columns.")
             return False
        newick_lookup = { (row['prune_name'], row['rgft_name']): row['newick'] for _, row in df_newick.iterrows() }
    except Exception as e:
        print(f"Error reading or processing newick file {newick_path}: {e}")
        return False

    # --- Pre-calculate features based on the input NNI tree ---
    print("Calculating base features from NNI tree...")
    features_nni_tree = calc_leaves_features(nni_tree_path, "prune")
    if features_nni_tree is None:
        print("Error: Failed to calculate base features from NNI tree. Aborting.")
        return False

    # --- Initialize New Feature Columns & Assign Group ID ---
    print("Initializing feature columns and assigning Group ID...")
    # Assign Group ID to the whole column at once to avoid dtype warnings
    df_summary[GROUP_ID_COL] = orig_ds_id
    # Initialize other feature columns
    all_feature_cols = list(FEATURES.values()) + list(RGFT_FEATURES.values()) + [LABEL]
    for feature_col in all_feature_cols:
        # Don't re-initialize group ID if it was already there
        if feature_col != GROUP_ID_COL and feature_col not in df_summary.columns:
            df_summary[feature_col] = np.nan
    # Ensure baseline LL columns are preserved
    if NNI_TREE_LL_COL not in df_summary.columns: df_summary[NNI_TREE_LL_COL] = np.nan
    if ORIG_DS_LL_COL not in df_summary.columns: df_summary[ORIG_DS_LL_COL] = np.nan


    # --- Prepare Cache ---
    remaining_tree_features_cache = {}

    print(f"Extracting features for {len(df_summary)} SPR moves...")
    # --- Iterate through SPR summary results and add features ---
    for idx, row in df_summary.iterrows():
        prune_name = row['prune_name']
        rgft_name = row['rgft_name']
        ll = row['ll']
        baseline_ll = row[ORIG_DS_LL_COL]

        # Group ID is already assigned

        neighbor_newick = newick_lookup.get((prune_name, rgft_name))
        if neighbor_newick is None: continue

        if prune_name not in remaining_tree_features_cache:
            subtree2_newick = newick_lookup.get((prune_name, SUBTREE2))
            if subtree2_newick is None: remaining_tree_features_cache[prune_name] = None
            else: remaining_tree_features_cache[prune_name] = calc_leaves_features(subtree2_newick, "rgft")
        features_rgft = remaining_tree_features_cache.get(prune_name)

        try:
            d_ll = float(ll) - float(baseline_ll)
            df_summary.at[idx, LABEL] = d_ll
        except (ValueError, TypeError):
            df_summary.at[idx, LABEL] = np.nan

        res_bl = calc_leaves_features(neighbor_newick, prune_name)
        df_summary.at[idx, FEATURES["res_bl"]] = res_bl

        # Assign NNI Tree Features (Prune Site) using .at
        df_summary.at[idx, FEATURES["bl"]] = features_nni_tree.get(FEATURES["bl"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["longest"]] = features_nni_tree.get(FEATURES["longest"], np.nan)
        df_summary.at[idx, FEATURES["ntaxa_p"]] = features_nni_tree.get(FEATURES["ntaxa_p"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["ntaxa_r"]] = features_nni_tree.get(FEATURES["ntaxa_r"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["tbl_p"]] = features_nni_tree.get(FEATURES["tbl_p"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["tbl_r"]] = features_nni_tree.get(FEATURES["tbl_r"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["longest_p"]] = features_nni_tree.get(FEATURES["longest_p"], {}).get(prune_name, np.nan)
        df_summary.at[idx, FEATURES["longest_r"]] = features_nni_tree.get(FEATURES["longest_r"], {}).get(prune_name, np.nan)

        # Assign Regraft Site Features using .at
        if features_rgft:
             df_summary.at[idx, RGFT_FEATURES["bl"]] = features_rgft.get(FEATURES["bl"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["ntaxa_p"]] = features_rgft.get(FEATURES["ntaxa_p"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["ntaxa_r"]] = features_rgft.get(FEATURES["ntaxa_r"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["tbl_p"]] = features_rgft.get(FEATURES["tbl_p"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["tbl_r"]] = features_rgft.get(FEATURES["tbl_r"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["longest_p"]] = features_rgft.get(FEATURES["longest_p"], {}).get(rgft_name, np.nan)
             df_summary.at[idx, RGFT_FEATURES["longest_r"]] = features_rgft.get(FEATURES["longest_r"], {}).get(rgft_name, np.nan)
        # else: Columns remain NaN

        # Assign Additional Regraft Features (Distances) using .at
        df_summary.at[idx, FEATURES["top_dist"]] = features_nni_tree.get(FEATURES["top_dist"], {}).get(prune_name, {}).get(rgft_name, np.nan)
        df_summary.at[idx, FEATURES["bl_dist"]] = features_nni_tree.get(FEATURES["bl_dist"], {}).get(prune_name, {}).get(rgft_name, np.nan)

    # --- Save Modified DataFrame back to original path ---
    print(f"Saving updated summary file with features to: {summary_path}")
    try:
        numeric_cols = ['ll', NNI_TREE_LL_COL, ORIG_DS_LL_COL, LABEL, FEATURES["res_bl"], FEATURES["bl_dist"]]
        numeric_cols.extend(list(RGFT_FEATURES.values()))
        numeric_cols.extend([FEATURES["bl"], FEATURES["longest"], FEATURES["ntaxa_p"], FEATURES["ntaxa_r"],
                             FEATURES["tbl_p"], FEATURES["tbl_r"], FEATURES["longest_p"], FEATURES["longest_r"]])
        for col in numeric_cols:
             if col in df_summary.columns:
                 df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
        # Ensure group ID is correct type before saving
        if GROUP_ID_COL in df_summary.columns:
             df_summary[GROUP_ID_COL] = df_summary[GROUP_ID_COL].astype(str) # Use string type

        df_summary.to_csv(summary_path, index=False)
        print("Successfully saved features by overwriting summary file.")
        return True
    except Exception as e:
        print(f"Error saving updated summary file to {summary_path}: {e}")
        return False
    finally:
        os.remove(newick_path) # Clean up newick file after processing
        # Note: The newick file is removed after processing to avoid cluttering the directory.


def get_input_trees(input_tree_file):
    """
    returns a list of input trees from the specified file.
    """
    if not os.path.exists(input_tree_file):
        print(f"Error: Input tree file not found at {input_tree_file}")
        return []

    df = pd.read_csv(input_tree_file)
    top_newick_trees = df['newick'].tolist()
    print(f"Found {len(top_newick_trees)} trees in {input_tree_file}.", flush=True)
    return top_newick_trees

def process_spr_results_wrapper(args):
    """
    Wrapper function to unpack arguments for run_spr_analysis.
    """
    return process_spr_results(*args)
# --- Main Execution Block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect features for SPR moves based on run_spr_on_tree.py outputs, adding features to the summary CSV.')
    parser.add_argument('--nni_tree_path', '-nni', required=True,
                        help='Path to the input NNI neighbor tree file (e.g., path/to/NNI/optimized_1.raxml.bestTree)')
    # parser.add_argument('--spr_output_prefix', '-p', required=True,
    #                     help='Path prefix for SPR output files, including base name but without suffix (e.g., path/to/output/optimized_1.raxml.bestTree)')
    parser.add_argument('--orig_ds_id', '-id', required=True, type=str,
                        help='Identifier for the original dataset this NNI tree belongs to (used for grouping in ML).')

    args = parser.parse_args()

    # --- Construct input paths from prefix ---
    nni_tree_path = args.nni_tree_path 
    dirname = os.path.dirname(nni_tree_path)

    trees = get_input_trees(nni_tree_path)

    # Prepare arguments for executor.map
    indexed_trees = [(tree, f"{dirname}/{str(i)}.spr_summary.csv", f"{dirname}/{str(i)}.spr_newicks.csv", args.orig_ds_id) for i, tree in enumerate(trees, start=1)]

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_spr_results_wrapper, indexed_trees))

    print(f"Finished run_spr_analysis on {nni_tree_path}.")

    # Exit with appropriate status code
    if any(result is False for result in results):
        print("One or more Feature Collection analyses failed. Check logs for details.")
        sys.exit(1)
    else:
        print("All Feature Collection completed successfully.")
    # Pass the summary_path as the file to be modified and the dataset ID
    # success = process_spr_results(nni_tree_path, summary_path, newick_path, args.orig_ds_id)

    # if success: 
    #     print("Feature collection completed successfully.")
    # else: 
    #     print("Feature collection encountered errors.")
    #     sys.exit(1)

