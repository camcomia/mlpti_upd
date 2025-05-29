import os, sys
import pandas as pd
import numpy as np
import argparse
from ete3 import PhyloTree # Or Tree, depending on what you use
from collections import OrderedDict

# Assuming defs_PhyAI.py is in the python path or same directory
# Import necessary variables and potentially helper functions if defined there

# --- Project-Specific Imports ---
SEP = "/"
ROOTLIKE_NAME = "ROOT_LIKE"
MSA_PHYLIP_FILENAME = "real_msa.phy"
PHYML_STATS_FILENAME = "real_msa.phy_phyml_stats_{0}.txt"
PHYML_TREE_FILENAME = "real_msa.phy_phyml_tree_{0}.txt"
SUMMARY_PER_DS = "{}ds_summary_{}_{}_step{}.csv"
PHYLIP_FORMAT = "phylip-relaxed"
TREES_PER_DS = "{}newicks_step{}.csv" # Define fallback
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"

# --- Helper functions (similar to those added to TBR.py) ---
def add_internal_names(t_orig):
    """Adds unique names (N0, N1, ...) to internal nodes if they lack names."""
    name_counter = 0
    existing_names = {node.name for node in t_orig.traverse() if node.name}
    nodes_named = 0
    for node in t_orig.traverse():
        if not node.is_leaf():
            if not node.name or node.name.isspace(): # Check for empty/whitespace names too
                new_name = f"N{name_counter}"
                while new_name in existing_names:
                    name_counter += 1
                    new_name = f"N{name_counter}"
                node.name = new_name
                existing_names.add(new_name)
                name_counter += 1
                nodes_named += 1
    # TODO set up loglevel to DEBUG
    # if nodes_named > 0:
    #     print(f"Assigned names to {nodes_named} internal nodes in collect_features.")
    return t_orig


def get_total_branch_length(tree):
    """Calculates the sum of all branch lengths in the tree."""
    if not tree: return 0
    return sum(node.dist for node in tree.traverse() if node.up)

def get_longest_branch(tree):
    """Finds the longest branch length in the tree."""
    if not tree: return 0
    return max([node.dist for node in tree.traverse() if node.up] + [0])

def get_edge_length_by_child_name(tree, child_name):
    """Finds the branch length leading to a node by its name."""
    # Debug log for tree and child name. TODO set up loglevel to DEBUG
    # print(f"Finding edge length in tree below for node: {child_name} \n {tree}")
    if not child_name: return np.nan # Handle missing names explicitly
    try:
        # Search for the node by name. Handle potential errors.
        nodes = tree.search_nodes(name=child_name)
        if nodes:
            node = nodes[0] # Take the first match if multiple exist (shouldn't happen with unique names)
            if node and node.up: # Check if it's not the root
                return node.dist
            elif node and node.is_root():
                 return 0 # Root has no incoming branch length by definition
        else:
            print(f"WARNING: Node '{child_name}' not found in tree for edge length.")
            sys.exit() # Suppress warning for potentially detached nodes during search
    except Exception as e:
        print(f"WARNING: Error finding node '{child_name}': {e}")
        sys.exit() # Suppress other potential errors
    return np.nan # Return NaN if not found or error

def get_distance_between_nodes(tree, name1, name2, topology_only=False):
    """Gets distance (branch length sum or topology) between two nodes by name."""
    if not name1 or not name2: return np.nan # Handle missing names
    try:
        node1 = tree & name1 # Use ETE3's direct search '&'
        node2 = tree & name2
        if node1 and node2:
             # Use topology_only=True for topological distance (number of nodes)
             # Default is branch length sum
            return node1.get_distance(node2, topology_only=topology_only)
        else:
            # print(f"WARNING: Node '{name1}' or '{name2}' not found for distance calc.")
            pass
    except Exception as e:
        # print(f"WARNING: Error getting distance between '{name1}' and '{name2}': {e}")
        pass
    return np.nan # Return NaN if not found or error

# --- Feature Calculation Logic ---

def calculate_remaining_tbr_features(row, t_orig):
    """
    Calculates the 8 features derivable from the original tree and move names.
    Takes a row from the input DataFrame and the original tree object.
    """
    features = {}
    bisect_node_name = row['bisect_branch']
    attach_node1_name = row['attach_edge1_node']
    attach_node2_name = row['attach_edge2_node']

    # Features 1 & 2: Original Tree Properties (calculated once outside the loop)
    # features['total_bl_orig'] = get_total_branch_length(t_orig) # Pass pre-calculated value
    # features['longest_bl_orig'] = get_longest_branch(t_orig) # Pass pre-calculated value

    # Feature 3: Branch length (bisection edge)
    features['bl_bisect_edge'] = get_edge_length_by_child_name(t_orig, bisect_node_name)

    # Feature 4 & 5: Branch lengths of insertion edges (in original tree)
    features['bl_insert_edge1'] = get_edge_length_by_child_name(t_orig, attach_node1_name)
    features['bl_insert_edge2'] = get_edge_length_by_child_name(t_orig, attach_node2_name)

    # Feature 6: Topological distance between insertion points (in original tree)
    features['topo_dist_insert'] = get_distance_between_nodes(t_orig, attach_node1_name, attach_node2_name, topology_only=True)

    # Feature 7: Branch length distance between insertion points (in original tree)
    bl_dist_insert = get_distance_between_nodes(t_orig, attach_node1_name, attach_node2_name, topology_only=False)
    features['bl_dist_insert'] = bl_dist_insert

    # Feature 8: Estimated new branch length (TBR) - Placeholder/Approximation
    # Using half the branch length distance between insertion points
    features['est_new_bl'] = bl_dist_insert / 2.0 if pd.notna(bl_dist_insert) else np.nan

    return pd.Series(features)


# --- Main Script Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect features for TBR rearrangements.')
    parser.add_argument('--dataset_path', '-ds', required=True, help='Path to the dataset directory.')
    # Add other arguments if needed (e.g., step number)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    step = 1 # Assuming step 1 for now, make this dynamic if needed

    # --- Define Input / Output Files ---
    # Input CSV from modified TBR.py
    input_csv_path = SUMMARY_PER_DS.format(dataset_path, "tbr_standard_with_sub_feats", "br", step)
    # Original tree file (ensure internal nodes have names)
    tree_file = os.path.join(dataset_path, PHYML_TREE_FILENAME.format("bionj"))
    # Output CSV for learning data
    output_csv_path = os.path.join(dataset_path, input_csv_path) # Prepend dataset path with input file from generating tbr moves

    # --- Load Data ---
    print(f"Loading TBR summary from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print(f"ERROR: Input CSV not found: {input_csv_path}")
        exit()
    try:
        tbr_summary_df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(tbr_summary_df)} TBR moves.")
    except Exception as e:
        print(f"ERROR: Failed to load input CSV: {e}")
        exit()

    print(f"Loading original tree from: {tree_file}")
    if not os.path.exists(tree_file):
        print(f"ERROR: Original tree file not found: {tree_file}")
        # Attempt to use the _no_internal version if get_tree logic isn't available/copied
        tree_file_no_internal = os.path.join(dataset_path, PHYML_TREE_FILENAME.format("bionj_no_internal"))
        if os.path.exists(tree_file_no_internal):
             print(f"WARNING: Using tree file without internal names: {tree_file_no_internal}. Ensure names are present if needed.")
             tree_file = tree_file_no_internal
        else:
             print(f"ERROR: No suitable tree file found.")
             exit()

    try:
        # Load tree using ete3. Ensure format=3 if names are in the file.
        # If get_tree function is available and handles naming, use it.
        # Otherwise, load directly:
        with open(tree_file, 'r') as f:
            newick_str = f.read().strip()
        
        t_orig = PhyloTree(newick_str, format=1) # Adjust format as needed. Use format=1 if simple newick, format=3 if ETE format with internal names
        # Ensure root has the expected name if calculations depend on it
        t_orig = add_internal_names(t_orig)
        t_orig.get_tree_root().name = ROOTLIKE_NAME
        print("Original tree loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load original tree: {e}")
        exit()

    # --- Calculate Original Tree Features (once) ---
    print("Calculating original tree features...")
    orig_total_bl = get_total_branch_length(t_orig)
    orig_longest_bl = get_longest_branch(t_orig)
    print(f"  Original Total BL: {orig_total_bl}")
    print(f"  Original Longest BL: {orig_longest_bl}")

    # --- Calculate Remaining Features for each move ---
    print("Calculating remaining features for each TBR move...")
    # Apply the function row-wise
    print(f"Showing columns in tbr_summary_df(length: {len(tbr_summary_df.columns)}): {tbr_summary_df.columns}")
    remaining_features_df = tbr_summary_df.apply(
        lambda row: calculate_remaining_tbr_features(row, t_orig),
        axis=1
    )
    print(f"Calculated {len(remaining_features_df.columns)} features: \t {remaining_features_df.columns}")

    # --- Combine DataFrames ---

    # Check if any calculated columns accidentally exist in the input summary and drop them
    cols_to_drop = [col for col in remaining_features_df.columns if col in tbr_summary_df.columns]
    if cols_to_drop:
        print(f"WARNING: Dropping potentially redundant columns from input summary: {cols_to_drop}")
        tbr_summary_df = tbr_summary_df.drop(columns=cols_to_drop)

    # Add the pre-calculated original tree features to remaining_features_df
    remaining_features_df['total_bl_orig'] = orig_total_bl
    remaining_features_df['longest_bl_orig'] = orig_longest_bl

    print(f"Combining features form summary.csv and calculated remaining features ({remaining_features_df.columns})...")
    # Concatenate the original summary info (excluding redundant subtree features) with the newly calculated features
    final_df = pd.concat([tbr_summary_df, remaining_features_df], axis=1)
    print(f"final_df columns with length of {len(final_df.columns)}: {final_df.columns}")

    # --- Define Final Feature Columns (using names consistent with previous discussion) ---
    # These are the 14 target features + identifiers + likelihood
    TBR_FEATURE_COLUMNS = [
        'total_bl_orig', 'longest_bl_orig', 'bl_bisect_edge',
        'bl_insert_edge1', 'bl_insert_edge2', 'topo_dist_insert',
        'bl_dist_insert', 'est_new_bl', 'leaves_sub1', 'total_bl_sub1',
        'longest_bl_sub1', 'leaves_sub2', 'total_bl_sub2', 'longest_bl_sub2'
    ]
    # Include move identifiers and target variable (likelihood difference)
    output_columns = ['move_id', 'bisect_branch', 'attach_edge1_node', 'attach_edge2_node', 'reconnect_type'] + TBR_FEATURE_COLUMNS + ['ll', 'orig_ds_ll'] # Add time if needed

    # Calculate likelihood difference (target variable)
    if 'll' in final_df.columns and 'orig_ds_ll' in final_df.columns:
         final_df['d_ll'] = final_df['ll'].astype(float) - final_df['orig_ds_ll'].astype(float)
         output_columns.append('d_ll')
    else:
         print("WARNING: 'll' or 'orig_ds_ll' columns missing, cannot calculate d_ll.")


    # Reorder columns and select the final set
    # Ensure all columns exist before selecting
    final_columns_present = [col for col in output_columns if col in final_df.columns]
    print(f"Final columns to save: {final_columns_present}")
    final_df = final_df[final_columns_present]

    # Add file path as orig_ds_id
    final_df['orig_ds_id'] = args.dataset_path;


    # --- Save Output ---
    print(f"Saving final features to: {output_csv_path}")
    try:
        final_df.to_csv(output_csv_path, index=False, na_rep='NaN') # Save NaN for missing values
        print("Feature collection complete.")
    except Exception as e:
        print(f"ERROR: Failed to save output CSV: {e}")

    newick_csv_file = TREES_PER_DS.format(dataset_path, "1") 
    try:
        os.remove(newick_csv_file) # Clean up newick file after processing
        print(f"Successfully deleted newicks csv file")
        # Note: The newick file is removed after processing to avoid cluttering the directory.
    except OSError as e: 
        print(f"Warning: Could not remove newick csv file {newick_csv_file}: {e}")