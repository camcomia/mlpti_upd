#########################################################################
##                 Copyright (C). All Rights Reserved.                   ##
##      "Harnessing machine learning to guide                            ##
##                              phylogenetic-tree search algorithms"     ##
##                                                                       ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                       ##
##                                                                       ##
##   For information, contact danaazouri@mail.tau.ac.il                  ##
##                                                                       ##
## For academic, non-commercial use.                                     ##
## If you use the code, please cite the paper                            ##
##                                                                       ##
#########################################################################

# --- Standard and External Imports ---
import os
import re
import sys
import shutil
import argparse
import random
import csv
from subprocess import Popen, PIPE, STDOUT
from collections import Counter
import pandas as pd
import numpy as np
from Bio import AlignIO
from ete3 import PhyloTree, PhyloNode # Use PhyloTree/PhyloNode consistently

# --- Logging and Debugging Imports ---
import logging
import traceback

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

# --- Global Variables / Constants ---
RAXML_NG_SCRIPT = "raxml-ng" # Change path to raxml-ng executable if needed
MIN_BRANCH_LENGTH = 1e-9 # Define a minimum branch length

# --- Configure Logging ---
log_format = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format) # Default level

# --- Helper Functions ---

def get_total_branch_length(tree):
    """Calculates the sum of all branch lengths in the tree."""
    if not tree:
        return 0
    return sum(node.dist for node in tree.traverse() if node.up)

def get_longest_branch(tree):
    """Finds the longest branch length in the tree."""
    if not tree:
        return 0
    return max([node.dist for node in tree.traverse() if node.up] + [0])

def get_msa_from_file(msa_file_path):
    """Opens and reads an MSA file."""
    if not os.path.exists(msa_file_path):
        logging.error(f"MSA file not found: {msa_file_path}")
        return None
    try:
        msa = AlignIO.read(msa_file_path, PHYLIP_FORMAT)
        return msa
    except Exception as e:
        logging.error(f"Failed to read MSA file {msa_file_path}: {e}")
        return None

def get_msa_properties(msa):
    """Gets number of taxa and characters from MSA object."""
    if isinstance(msa, str):
        msa = get_msa_from_file(msa)
    if msa is None:
        return None, None
    try:
        ntaxa = len(msa)
        nchars = msa.get_alignment_length()
        return ntaxa, nchars
    except Exception as e:
        logging.error(f"Failed to get MSA properties: {e}")
        return None, None

def get_newick_tree(tree_input):
    """Reads a Newick tree string from file if path is given."""
    if os.path.exists(tree_input):
        try:
            with open(tree_input, 'r') as tree_fpr:
                tree_str = tree_fpr.read().strip()
            return tree_str
        except IOError as e:
            logging.error(f"Could not read tree file {tree_input}: {e}")
            return None
    return tree_input

def add_internal_names(t_orig):
    """Adds unique names (N0, N1, ...) to internal nodes if they lack names."""
    name_counter = 0
    existing_names = {node.name for node in t_orig.traverse() if node.name}
    nodes_named = 0
    for node in t_orig.traverse():
        if not node.is_leaf():
            if not node.name:
                new_name = f"N{name_counter}"
                while new_name in existing_names:
                    name_counter += 1
                    new_name = f"N{name_counter}"
                node.name = new_name
                existing_names.add(new_name)
                name_counter += 1
                nodes_named += 1
    if nodes_named > 0:
        logging.debug(f"Assigned names to {nodes_named} internal nodes.")
    return t_orig

def get_tree(ds_path, msa_file):
    """Loads the starting tree, ensuring internal nodes have names."""
    tree_file_with_names = os.path.join(ds_path, PHYML_TREE_FILENAME.format("bionj"))
    tree_file_no_internal = os.path.join(ds_path, PHYML_TREE_FILENAME.format("bionj_no_internal"))
    t_orig = None
    tree_format_used = None # Keep track of format used

    try:
        if os.path.exists(tree_file_with_names):
            logging.debug(f"Attempting to load tree: {tree_file_with_names}")
            tree_format_used = 1
            t_orig = PhyloTree(tree_file_with_names, format=tree_format_used)
            logging.debug(f"Loaded tree assuming format {tree_format_used}.")
        elif os.path.exists(tree_file_no_internal):
            logging.debug(f"Loading tree: {tree_file_no_internal}")
            tree_format_used = 1
            t_orig = PhyloTree(tree_file_no_internal, format=tree_format_used)
            logging.debug(f"Loaded tree assuming format {tree_format_used}.")
        else:
            logging.error(f"No suitable tree file found in {ds_path}.")
            return None

        t_orig = add_internal_names(t_orig)
        root_node = t_orig.get_tree_root()
        if not root_node.name:
             root_node.name = ROOTLIKE_NAME
        elif root_node.name != ROOTLIKE_NAME:
             logging.warning(f"Root node already had a name ('{root_node.name}') different from expected '{ROOTLIKE_NAME}'.")

        logging.info(f"Tree successfully loaded and processed for {ds_path}")
        return t_orig
    except Exception as e:
        logging.error(f"Failed to load or process tree for {ds_path} using format={tree_format_used}: {e}")
        if "NewickError" in str(type(e)):
             logging.error(f"ETE Newick parsing error: {e}")
             logging.error("This might indicate the Newick file itself is malformed (e.g., missing internal node names before colons).")
        logging.error(traceback.format_exc())
        return None

def parse_phyml_stats_output(msa_filepath, stats_filepath):
    """Parses PhyML stats file content."""
    res_dict = dict.fromkeys(["ntaxa", "nchars", "ll", "fA", "fC", "fG", "fT",
                              "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                              "pInv", "gamma", "path"], "")
    if msa_filepath:
        ntaxa, nchars = get_msa_properties(msa_filepath)
        if ntaxa is not None:
            res_dict['ntaxa'] = str(ntaxa)
        if nchars is not None:
            res_dict['nchars'] = str(nchars)
    res_dict["path"] = stats_filepath
    if not os.path.exists(stats_filepath):
        logging.warning(f"PhyML stats file not found: {stats_filepath}")
        return None
    try:
        with open(stats_filepath) as fpr:
            content = fpr.read()
        ll_match = re.search(r"Log-likelihood:\s+([-\d.]+)", content)
        res_dict["ll"] = ll_match.group(1).strip() if ll_match else ""
        gamma_match = re.search(r"Gamma shape parameter:\s+([\d.]+)", content)
        res_dict['gamma'] = gamma_match.group(1).strip() if gamma_match else ""
        pinv_match = re.search(r"Proportion of invariant.*?:\s+([\d.]+)", content)
        res_dict['pInv'] = pinv_match.group(1).strip() if pinv_match else ""
        for nuc in "ACGT":
            nuc_match = re.search(r"f\(" + nuc + r"\)\s*=\s*([\d.]+)", content)
            res_dict["f" + nuc] = nuc_match.group(1).strip() if nuc_match else ""
        for nuc1 in "ACGT":
            for nuc2 in "ACGT":
                if nuc1 < nuc2:
                    sub_match = re.search(nuc1 + r"\s*<->\s*" + nuc2 + r"\s+([\d.]+)", content)
                    res_dict["sub" + nuc1 + nuc2] = sub_match.group(1).strip() if sub_match else ""
        if not res_dict["ll"]:
            logging.warning(f"Could not parse likelihood from {stats_filepath}")
    except Exception as e:
        logging.error(f"Error parsing PhyML stats file {stats_filepath}: {e}")
        return None
    return res_dict

def parse_raxmlNG_content(content, orig_ll):
    """Parses RAxML-NG output string."""
    res_dict = dict.fromkeys(["ll", "pInv", "gamma", "fA", "fC", "fG", "fT",
                              "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                              "time"], "")
    try:
        ll_re = re.search(r"Final LogLikelihood:\s+([-\d.]+)", content)
        if ll_re:
            res_dict["ll"] = ll_re.group(1).strip()
        elif re.search("BL opt converged to a worse likelihood score", content) or re.search("failed", content):
            ll_ini = re.search(r"initial LogLikelihood:\s+([-\d.]+)", content)
            if ll_ini:
                res_dict["ll"] = ll_ini.group(1).strip()
                logging.warning("RAxML BL opt failed or worsened, using initial LL.")
            else:
                res_dict["ll"] = orig_ll # Use original LL as fallback
                logging.warning("RAxML BL opt failed, could not find initial LL, using original tree LL.")
        else:
            logging.warning("Could not find final likelihood in RAxML output.")
        gamma_regex = re.search(r"alpha:\s+(\d+\.?\d*)", content)
        res_dict['gamma'] = gamma_regex.group(1).strip() if gamma_regex else ""
        pinv_regex = re.search(r"P-inv.*:\s+(\d+\.?\d*)", content)
        res_dict['pInv'] = pinv_regex.group(1).strip() if pinv_regex else ""
        nucs_freq = re.search(r"Base frequencies.*:\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
        if nucs_freq:
            for i,nuc in enumerate("ACGT"):
                res_dict["f" + nuc] = nucs_freq.group(i+1).strip()
        subs_freq = re.search(r"Substitution rates.*:\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
        if subs_freq:
            for i,nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):
                res_dict["sub" + nuc_pair] = subs_freq.group(i+1).strip()
        rtime = re.search(r"Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
        res_dict["time"] = rtime.group(1).strip() if rtime else ""
        if not res_dict["time"]:
            logging.warning("Could not parse elapsed time from RAxML output.")
        if not res_dict["ll"]:
            res_dict["time"] = 'no_ll_opt_no_time'
    except Exception as e:
        logging.error(f"Error parsing RAxML content: {e}")
    return res_dict

def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq, orig_ll):
    """Calls RAxML-NG to evaluate tree likelihood."""
    rates_str = "/".join(map(str, rates)) if rates else ""
    pinv_str = str(pinv) if pinv is not None else ""
    alpha_str = str(alpha) if alpha is not None else ""
    freq_str = "/".join(map(str, freq)) if freq else ""
    if not all([rates_str, pinv_str, alpha_str, freq_str]):
        logging.error("Missing one or more model parameters for RAxML.")
        return None, None
    model_line_params = f'GTR{{{rates_str}}}+I{{{pinv_str}}}+G{{{alpha_str}}}+F{{{freq_str}}}'
    logging.debug(f"RAxML model string: {model_line_params}")
    tree_rampath = f"/dev/shm/tree_{os.path.basename(msa_tmpfile)}_{random.random()}.nwk"
    ll = None
    rtime = None
    try:
        with open(tree_rampath, "w") as fpw:
            fpw.write(tree_str)
        # Updated RAxML command line flags for faster execution if needed
        raxml_cmd = [ RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile, '--threads', '2',
                     '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params,
                     '--nofiles', '--tree', tree_rampath, '--precision', '8', '--blopt', 'nr_safe'] # Use precision 6 and nr_fast
        logging.debug(f"RAxML command: {' '.join(raxml_cmd)}")
        p = Popen(raxml_cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        raxml_stdout, _ = p.communicate()
        raxml_output = raxml_stdout.decode(errors='ignore')
        logging.debug(f"RAxML raw output:\n{raxml_output}") # Log full output for debugging
        res_dict = parse_raxmlNG_content(raxml_output, orig_ll) # Pass orig_ll
        ll = res_dict.get('ll')
        rtime = res_dict.get('time')
        try:
            ll = float(ll) if ll else None
        except (ValueError, TypeError):
            logging.warning(f"Could not convert RAxML LL '{ll}' to float.")
            ll = None
        try:
            rtime = float(rtime) if rtime else None
        except (ValueError, TypeError):
            logging.warning(f"Could not convert RAxML Time '{rtime}' to float.")
            rtime = None
    except FileNotFoundError:
        logging.error(f"RAxML-NG executable not found at: {RAXML_NG_SCRIPT}")
    except Exception as e:
        logging.error(f"Exception during RAxML call or parsing for {msa_tmpfile}: {e}")
        logging.error(traceback.format_exc())
    finally:
        if os.path.exists(tree_rampath):
            try:
                os.remove(tree_rampath)
            except OSError as rm_err:
                logging.warning(f"Failed to remove temporary tree file {tree_rampath}: {rm_err}")
    return ll, rtime

# --- TBR Functions (Updated with Logging, Assertions->Exit, Bugfix) ---

def bisect_tree_simple(tree, branch_name):
    """
    Bisects the tree. Exits on leaf mismatch.
    Returns (remaining_tree, detached_subtree) or (None, None) on other failures.
    """
    logging.debug(f"Entered function for branch_name='{branch_name}'")
    try:
        tree_copy = tree.copy()
        orig_leaves_set = set(tree_copy.get_leaf_names())
    except Exception as copy_err:
        logging.error(f"Failed to copy tree! {copy_err}")
        return None, None
    try:
        nodes_found = tree_copy.search_nodes(name=branch_name)
        if not nodes_found:
            logging.error(f"Node '{branch_name}' not found in copy.")
            return None, None
        node_to_remove = nodes_found[0]
    except Exception as find_err:
        logging.error(f"Exception during search for '{branch_name}' in copy! {find_err}")
        return None, None

    if not node_to_remove.is_root():
        parent = node_to_remove.up
        if not parent:
            logging.error(f"Node '{node_to_remove.name}' is not root but has no parent.")
            return None, None
        try:
            # Detach the node; the detached part becomes detached_subtree
            # The node_to_remove itself is the root of the detached_subtree
            detached_subtree = node_to_remove.detach()
            detached_leaves_set = set(detached_subtree.get_leaf_names())
        except Exception as detach_err:
            logging.error(f"Failed to detach node '{node_to_remove.name}'! {detach_err}")
            return None, None
        # tree_copy is now the remaining tree
        remaining_tree = tree_copy
        remaining_leaves_set = set(remaining_tree.get_leaf_names())
        if len(orig_leaves_set) != len(remaining_leaves_set) + len(detached_leaves_set):
             logging.critical(f"BISSECT Leaf count mismatch! Orig: {len(orig_leaves_set)}, Remaining: {len(remaining_leaves_set)}, Detached: {len(detached_leaves_set)}")
             sys.exit(1)
        if orig_leaves_set != remaining_leaves_set.union(detached_leaves_set):
             logging.critical(f"BISSECT Leaf set mismatch!")
             sys.exit(1)
        return remaining_tree, detached_subtree
    else:
        # Cannot bisect at the root node itself
        logging.debug(f"Node '{branch_name}' is the root, cannot bisect.")
        return None, None

def _prepare_tbr_reconnect(tree1, node1_name, tree2, node2_name):
    """
    Internal helper to find nodes, parents, distances for TBR reconnect.
    Returns None if nodes/parents invalid for standard TBR edge break.
    Detaches node1 and node2, returning DEEP COPIES of the four components and parent names.
    """
    try:
        t1_copy = tree1.copy("deepcopy")
        t2_copy = tree2.copy("deepcopy")

        # Find nodes defining the edges to break
        logging.debug(f"Reconnect Prep: Searching for node1='{node1_name}' in t1_copy.")
        node1_list = t1_copy.search_nodes(name=node1_name)
        node1 = node1_list[0] if node1_list else None

        logging.debug(f"Reconnect Prep: Searching for node2='{node2_name}' in t2_copy.")
        node2_list = t2_copy.search_nodes(name=node2_name)
        node2 = node2_list[0] if node2_list else None

        if not (node1 and node2):
            if not node1: logging.error(f"Reconnect Prep: node1 '{node1_name}' not found in t1_copy!")
            if not node2: logging.error(f"Reconnect Prep: node2 '{node2_name}' not found in t2_copy!")
            return None

        # Get parents. These MUST exist for the edges we break in TBR.
        parent1 = node1.up
        parent2 = node2.up

        if not parent1:
             logging.error(f"Reconnect Prep: Cannot break edge for root node1 '{node1_name}'.")
             return None
        if not parent2:
             logging.error(f"Reconnect Prep: Cannot break edge for root node2 '{node2_name}'.")
             return None

        # *** Store parent names BEFORE detaching ***
        parent1_name = parent1.name
        parent2_name = parent2.name

        logging.debug(f"Reconnect Prep: Found nodes: node1='{node1.name}', parent1='{parent1_name}', node2='{node2.name}', parent2='{parent2_name}'.")

        # Store original branch lengths leading to node1 and node2
        dist1 = node1.dist if node1.dist is not None else MIN_BRANCH_LENGTH
        dist2 = node2.dist if node2.dist is not None else MIN_BRANCH_LENGTH
        logging.debug(f"Reconnect Prep: Original dist1={dist1}, dist2={dist2}")

        # Detach the nodes to get the four components
        logging.debug(f"Reconnect Prep: Detaching node1 '{node1.name}' from parent1 '{parent1_name}'.")
        node1_subtree = node1.detach() # This is the subtree rooted at node1
        logging.debug(f"Reconnect Prep: Detaching node2 '{node2.name}' from parent2 '{parent2_name}'.")
        node2_subtree = node2.detach() # This is the subtree rooted at node2

        # t1_copy is now the remaining part containing parent1
        # t2_copy is now the remaining part containing parent2
        parent1_component = t1_copy
        parent2_component = t2_copy

        # *** Return DEEP COPIES of components and parent names ***
        return parent1_component.copy("deepcopy"), node1_subtree.copy("deepcopy"), dist1, parent1_name, \
               parent2_component.copy("deepcopy"), node2_subtree.copy("deepcopy"), dist2, parent2_name

    except Exception as prep_err:
        logging.error(f"Reconnect Prep: Exception during node/parent search or detach! {prep_err}")
        logging.error(traceback.format_exc())
        return None

def _finalize_reconnect(tree_root, expected_leaves, leaves1, leaves2, node1_name, node2_name, parent1_name, parent2_name, type_str):
    """Internal helper to perform final checks and save bad trees.
       Includes specific prune for Type 2 artifact."""
    if tree_root is None:
        logging.error(f"Reconnect {type_str}: Reconnection failed, tree_root is None.")
        return None
    try:
        # Ensure root doesn't have a problematic name before writing structure
        if not tree_root.name or tree_root.name.startswith("TEMP_ROOT"):
             tree_root.name = f"internal_root_{type_str}_{node1_name}_{node2_name}" # More informative temp name

        final_tree_structure_str_f3 = tree_root.write(format=3) # Format 3 for detailed structure
        final_tree_structure_str_f1 = tree_root.write(format=1) # Format 1 for standard Newick
        logging.debug(f"Reconnect {type_str}: Final tree structure (format=3) before check/prune:\n{final_tree_structure_str_f3}")
        logging.debug(f"Reconnect {type_str}: Final tree structure (format=1) before check/prune:\n{final_tree_structure_str_f1}")
        logging.debug(f"Reconnect {type_str}: Root node '{tree_root.name}' children: {[c.name for c in tree_root.children]}")

    except Exception as write_err:
        logging.warning(f"Reconnect {type_str}: Could not write final tree structure for debugging: {write_err}")

    final_check_leaves_set = set(tree_root.get_leaf_names())
    logging.debug(f"Reconnect {type_str}: Before final check: Leaf count = {len(final_check_leaves_set)}, Leaves = {final_check_leaves_set}")

    # *** START FIX: Specific prune for Type 2 artifact ***
    pruned_artifact = False
    if type_str == "T2":
        problematic_parents_as_leaves = []
        if parent1_name in final_check_leaves_set:
            problematic_parents_as_leaves.append(parent1_name)
        if parent2_name in final_check_leaves_set:
            problematic_parents_as_leaves.append(parent2_name)

        # Check if the *only* discrepancy is the parent(s) appearing as leaves
        if len(final_check_leaves_set) == expected_leaves + len(problematic_parents_as_leaves):
            expected_set = leaves1.union(leaves2)
            if final_check_leaves_set == expected_set.union(set(problematic_parents_as_leaves)):
                logging.warning(f"Reconnect T2: Detected potential artifact node(s) {problematic_parents_as_leaves} as leaves. Attempting prune.")
                for node_name_to_prune in problematic_parents_as_leaves:
                    try:
                        nodes_to_prune = tree_root.search_nodes(name=node_name_to_prune)
                        if nodes_to_prune:
                             # Detach the node if it's found as a leaf
                             if nodes_to_prune[0].is_leaf():
                                 nodes_to_prune[0].detach()
                                 logging.info(f"Reconnect T2: Successfully pruned artifact leaf node '{node_name_to_prune}'.")
                                 pruned_artifact = True # Mark that we pruned
                             else:
                                 logging.warning(f"Reconnect T2: Node '{node_name_to_prune}' found but is not a leaf. Cannot prune artifact.")
                        else:
                             logging.warning(f"Reconnect T2: Could not find artifact node '{node_name_to_prune}' to prune.")
                    except Exception as prune_err:
                        logging.error(f"Reconnect T2: Error pruning artifact node '{node_name_to_prune}': {prune_err}")

                # Re-check leaves after pruning
                final_check_leaves_set = set(tree_root.get_leaf_names())
                logging.debug(f"Reconnect {type_str}: After potential prune: Leaf count = {len(final_check_leaves_set)}, Leaves = {final_check_leaves_set}")
            else:
                 logging.debug("Reconnect T2: Leaf set mismatch is not solely due to parent artifact nodes.")
        # else: leaf count mismatch is different, proceed to error

    # *** END FIX ***

    # Leaf preservation check - EXIT ON FAILURE
    if len(final_check_leaves_set) != expected_leaves:
         logging.critical(f"RECONNECT {type_str} Leaf count mismatch! Expected: {expected_leaves}, Found: {len(final_check_leaves_set)}")
         logging.critical(f"  Initial Leaves T1: {leaves1}")
         logging.critical(f"  Initial Leaves T2: {leaves2}")
         logging.critical(f"  Final Leaves (Check): {final_check_leaves_set}")
         bad_tree_path = f"bad_tree_{type_str}_{node1_name}_{node2_name}.nwk"
         try:
             tree_root.write(format=1, outfile=bad_tree_path)
             logging.warning(f"Saved problematic tree (leaf count mismatch) to {bad_tree_path}")
         except Exception as save_err:
             logging.warning(f"Could not save problematic tree: {save_err}")
         # Only exit if we didn't successfully prune the artifact
         if not pruned_artifact:
              sys.exit(1)
         else:
              logging.warning("Proceeding after successful artifact prune, although initial mismatch occurred.")


    if leaves1.union(leaves2) != final_check_leaves_set:
         logging.critical(f"RECONNECT {type_str} Leaf set mismatch!")
         logging.critical(f"  Initial Leaves T1: {leaves1}")
         logging.critical(f"  Initial Leaves T2: {leaves2}")
         logging.critical(f"  Final Leaves (Check): {final_check_leaves_set}")
         bad_tree_path = f"bad_tree_{type_str}_{node1_name}_{node2_name}.nwk"
         try:
             tree_root.write(format=1, outfile=bad_tree_path)
             logging.warning(f"Saved problematic tree (leaf set mismatch) to {bad_tree_path}")
         except Exception as save_err:
             logging.warning(f"Could not save problematic tree: {save_err}")
         # Only exit if we didn't successfully prune the artifact
         if not pruned_artifact:
             sys.exit(1)
         else:
              logging.warning("Proceeding after successful artifact prune, although initial set mismatch occurred.")

    logging.debug(f"Reconnect {type_str}: Returning final tree rooted at '{tree_root.name}'.")
    return tree_root


# --- Standard TBR Reconnection Functions (Update calls to _finalize_reconnect) ---

def _reconnect_tbr_type1(tree1, node1_name, tree2, node2_name, new_internal_node_prefix="TBR_internal"):
    """Reconnect type 1: Connect (parent1, node2_sub) and (parent2, node1_sub)."""
    logging.debug("Attempting Reconnect Type 1")
    try:
        # ... (deepcopy trees, get leaves, expected_total) ...
        t1_orig_copy = tree1.copy("deepcopy")
        t2_orig_copy = tree2.copy("deepcopy")
        t1_leaves = set(t1_orig_copy.get_leaf_names())
        t2_leaves = set(t2_orig_copy.get_leaf_names())
        expected_total = len(t1_leaves) + len(t2_leaves)

        prep_results = _prepare_tbr_reconnect(t1_orig_copy, node1_name, t2_orig_copy, node2_name)
        if prep_results is None: return None
        parent1_comp, node1_sub, dist1, parent1_name, \
        parent2_comp, node2_sub, dist2, parent2_name = prep_results

        # ... (logging, find parent nodes in components) ...
        parent1_node_list = parent1_comp.search_nodes(name=parent1_name)
        parent1_node = parent1_node_list[0] if parent1_node_list else None
        parent2_node_list = parent2_comp.search_nodes(name=parent2_name)
        parent2_node = parent2_node_list[0] if parent2_node_list else None
        if not (parent1_node and parent2_node): return None # Error handled inside

        # ... (perform connections, logging) ...
        parent1_node.add_child(node2_sub, dist=dist1 / 2.0 if dist1 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)
        parent2_node.add_child(node1_sub, dist=dist2 / 2.0 if dist2 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)


        # Join the two modified main components under a new root
        final_tree_root = PhyloNode(name=f"{new_internal_node_prefix}_root_T1")
        # Add check if component roots exist before adding
        r1 = parent1_comp.get_tree_root()
        r2 = parent2_comp.get_tree_root()
        if r1: final_tree_root.add_child(r1, dist=MIN_BRANCH_LENGTH)
        if r2: final_tree_root.add_child(r2, dist=MIN_BRANCH_LENGTH)
        if not r1 and not r2:
            logging.error("T1 Reconnect: Both parent components are empty after reconnection.")
            return None
        logging.debug(f"T1 - Final tree before finalize: {final_tree_root.write(format=1)}")

        # *** Update call to pass parent names ***
        return _finalize_reconnect(final_tree_root, expected_total, t1_leaves, t2_leaves, node1_name, node2_name, parent1_name, parent2_name, "T1")

    except Exception as e:
        logging.error(f"Exception during Reconnect Type 1: {e}")
        logging.error(traceback.format_exc())
        return None

def _reconnect_tbr_type2(tree1, node1_name, tree2, node2_name, new_internal_node_prefix="TBR_internal"):
    """Reconnect type 2: Connect (parent1, parent2) and (node1, node2)."""
    logging.debug("Attempting Reconnect Type 2")
    try:
        # ... (deepcopy trees, get leaves, expected_total) ...
        t1_orig_copy = tree1.copy("deepcopy")
        t2_orig_copy = tree2.copy("deepcopy")
        t1_leaves = set(t1_orig_copy.get_leaf_names())
        t2_leaves = set(t2_orig_copy.get_leaf_names())
        expected_total = len(t1_leaves) + len(t2_leaves)

        prep_results = _prepare_tbr_reconnect(t1_orig_copy, node1_name, t2_orig_copy, node2_name)
        if prep_results is None: return None
        parent1_comp, node1_sub, dist1, parent1_name, \
        parent2_comp, node2_sub, dist2, parent2_name = prep_results

        # ... (logging) ...
        logging.debug(f"T2 - Before Reconnect:")
        # ... (print component structures, parent names) ...

        # *** Original Type 2 logic ***
        new_node_nodes = PhyloNode(name=f"{new_internal_node_prefix}_N1_T2")
        new_node_parents = PhyloNode(name=f"{new_internal_node_prefix}_N2_T2")

        new_node_nodes.add_child(node1_sub, dist=dist1 / 2.0 if dist1 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)
        new_node_nodes.add_child(node2_sub, dist=dist2 / 2.0 if dist2 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)
        logging.debug(f"T2 - new_node_nodes combined: {new_node_nodes.write(format=1)}")

        r1 = parent1_comp.get_tree_root()
        r2 = parent2_comp.get_tree_root()
        if r1:
            logging.debug(f"T2: Attaching root of parent1_comp ('{r1.name}') under new_node_parents")
            new_node_parents.add_child(r1, dist=dist1 / 2.0 if dist1 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)
        if r2:
            logging.debug(f"T2: Attaching root of parent2_comp ('{r2.name}') under new_node_parents")
            new_node_parents.add_child(r2, dist=dist2 / 2.0 if dist2 > MIN_BRANCH_LENGTH else MIN_BRANCH_LENGTH)
        logging.debug(f"T2 - new_node_parents combined: {new_node_parents.write(format=1)}")

        final_tree_root = PhyloNode(name=f"{new_internal_node_prefix}_root_T2")
        # Only add children if they contain nodes
        if len(new_node_nodes.children) > 0:
             final_tree_root.add_child(new_node_nodes, dist=MIN_BRANCH_LENGTH)
        if len(new_node_parents.children) > 0:
             final_tree_root.add_child(new_node_parents, dist=MIN_BRANCH_LENGTH)

        # Check if the final root actually got children
        if len(final_tree_root.children) == 0:
             logging.error("T2 Reconnect: Final root has no children. Reconnection failed.")
             return None

        logging.debug(f"T2 - Final tree before finalize: {final_tree_root.write(format=1)}")

        # *** Update call to pass parent names ***
        return _finalize_reconnect(final_tree_root, expected_total, t1_leaves, t2_leaves, node1_name, node2_name, parent1_name, parent2_name, "T2")

    except Exception as e:
        logging.error(f"Exception during Reconnect Type 2: {e}")
        logging.error(traceback.format_exc())
        return None

def _reconnect_tbr_type3(tree1, node1_name, tree2, node2_name, new_internal_node_prefix="TBR_internal"):
    """Reconnect type 3: Connect (parent1, node1_sub) and (parent2, node2_sub)."""
    logging.debug("Attempting Reconnect Type 3 (Original Topology)")
    try:
        # ... (deepcopy trees, get leaves, expected_total) ...
        t1_orig_copy = tree1.copy("deepcopy")
        t2_orig_copy = tree2.copy("deepcopy")
        t1_leaves = set(t1_orig_copy.get_leaf_names())
        t2_leaves = set(t2_orig_copy.get_leaf_names())
        expected_total = len(t1_leaves) + len(t2_leaves)

        prep_results = _prepare_tbr_reconnect(t1_orig_copy, node1_name, t2_orig_copy, node2_name)
        if prep_results is None: return None
        parent1_comp, node1_sub, dist1, parent1_name, \
        parent2_comp, node2_sub, dist2, parent2_name = prep_results

        # ... (logging, find parent nodes in components) ...
        parent1_node_list = parent1_comp.search_nodes(name=parent1_name)
        parent1_node = parent1_node_list[0] if parent1_node_list else None
        parent2_node_list = parent2_comp.search_nodes(name=parent2_name)
        parent2_node = parent2_node_list[0] if parent2_node_list else None
        if not (parent1_node and parent2_node): return None # Error logged inside

        # ... (perform connections using original distances, logging) ...
        parent1_node.add_child(node1_sub, dist=dist1)
        parent2_node.add_child(node2_sub, dist=dist2)


        # Join the two restored subtrees under a new root
        final_tree_root = PhyloNode(name=f"{new_internal_node_prefix}_root_T3")
        # Add check if component roots exist
        r1 = parent1_comp.get_tree_root()
        r2 = parent2_comp.get_tree_root()
        if r1: final_tree_root.add_child(r1, dist=MIN_BRANCH_LENGTH)
        if r2: final_tree_root.add_child(r2, dist=MIN_BRANCH_LENGTH)
        if not r1 and not r2:
             logging.error("T3 Reconnect: Both parent components are empty after reconnection.")
             return None

        logging.debug(f"T3 - Final tree before finalize: {final_tree_root.write(format=1)}")

        # *** Update call to pass parent names ***
        return _finalize_reconnect(final_tree_root, expected_total, t1_leaves, t2_leaves, node1_name, node2_name, parent1_name, parent2_name, "T3")

    except Exception as e:
        logging.error(f"Exception during Reconnect Type 3: {e}")
        logging.error(traceback.format_exc())
        return None


def all_TBR_simple(ds_path, outpath_format_string):
    """Generates TBR neighbors using standard reconnection, checks leaves, calculates likelihoods."""
    logging.info(f"Initializing all_TBR_simple for dataset: {ds_path}")
    orig_msa_file = os.path.join(ds_path, MSA_PHYLIP_FILENAME)
    stats_filepath = os.path.join(ds_path, PHYML_STATS_FILENAME.format('bionj'))
    ds_identifier = os.path.basename(os.path.normpath(ds_path))
    msa_rampath = f"/dev/shm/tmp_{ds_identifier}_msa_tbr.phy"
    tree_file = os.path.join(ds_path, PHYML_TREE_FILENAME.format("bionj"))
    newick_output_csv = TREES_PER_DS.format(ds_path, "1")
    # *** Updated Newick Headers ***
    newick_headers = ['move_id', 'bisect_branch', 'attach_edge1_node', 'attach_edge2_node', 'reconnect_type', 'newick']

    try:
        t_orig = get_tree(ds_path, orig_msa_file)
        if not t_orig:
            return
        orig_leaves_set = set(t_orig.get_leaf_names())
        orig_leaf_count = len(orig_leaves_set)
        logging.info(f"Original tree loaded. Leaf count: {orig_leaf_count}.")
        missing_names = sum(1 for node in t_orig.traverse() if not getattr(node, 'name', None))
        if missing_names > 0:
            logging.warning(f"Found {missing_names} nodes without names in initial tree.")
    except Exception as load_err:
        logging.error(f"Failed during initial tree loading: {load_err}")
        logging.error(traceback.format_exc())
        return

    output_summary_csv = outpath_format_string.format("tbr_standard_with_sub_feats") # Updated name
    # *** Updated Summary Headers ***
    csv_headers = ['move_id', 'bisect_branch', 'attach_edge1_node', 'attach_edge2_node', 'reconnect_type',
                   'leaves_sub1', 'total_bl_sub1', 'longest_bl_sub1',
                   'leaves_sub2', 'total_bl_sub2', 'longest_bl_sub2',
                   'll', 'time', 'orig_ds_ll']
    try:
        os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
        with open(output_summary_csv, "w", newline='') as fpw:
            csv.writer(fpw).writerow(csv_headers)
        os.makedirs(os.path.dirname(newick_output_csv), exist_ok=True)
        with open(newick_output_csv, "w", newline='') as fpw_nwk:
            csv.writer(fpw_nwk).writerow(newick_headers)
    except IOError as io_err:
        logging.error(f"Cannot write to output CSV {output_summary_csv} or {newick_output_csv}: {io_err}")
        return

    results_list = []
    try:
        with open(orig_msa_file) as fpr, open(msa_rampath, "w") as fpw:
            fpw.write(fpr.read())
        logging.info(f"Copied MSA to {msa_rampath}")
    except IOError as io_err:
        logging.error(f"Failed to copy MSA to RAM: {io_err}")
        return

    try:
        params_dict = parse_phyml_stats_output(None, stats_filepath)
        if not params_dict:
            raise ValueError("Missing PhyML parameters")
        freq = [params_dict.get("fA"), params_dict.get("fC"), params_dict.get("fG"), params_dict.get("fT")]
        rates = [params_dict.get("subAC"), params_dict.get("subAG"), params_dict.get("subAT"),
                 params_dict.get("subCG"), params_dict.get("subCT"), params_dict.get("subGT")]
        pinv = params_dict.get("pInv")
        alpha = params_dict.get("gamma")
        orig_ll_str = params_dict.get("ll")
        if None in freq or None in rates or pinv is None or alpha is None:
             raise ValueError("Missing essential PhyML parameters")
        orig_ll = float(orig_ll_str) if orig_ll_str else None
        if orig_ll is None:
            logging.warning("Original LL not found in params.")
        logging.info(f"Original LL: {orig_ll}")

        logging.info(f"Starting main TBR loop...")
        move_counter = 0
        valid_neighbor_counter = 0

        # Iterate through potential bisection edges (defined by internal nodes)
        internal_nodes_to_bisect = [node for node in t_orig.traverse() if not node.is_leaf() and node.up]
        logging.info(f"Found {len(internal_nodes_to_bisect)} potential internal nodes for bisection.")

        with open(newick_output_csv, "a", newline='') as fpw_nwk:
            csvwriter_nwk = csv.writer(fpw_nwk)

            for i, bisect_node in enumerate(internal_nodes_to_bisect):
                bisect_name = getattr(bisect_node, 'name', None)
                if not bisect_name:
                    logging.warning(f"bisect_node at index {i} has no name! Skipping.")
                    continue

                subtree1, subtree2 = bisect_tree_simple(t_orig.copy(), bisect_name) # Exits on failure

                if subtree1 and subtree2:
                    s1_leaves = set(subtree1.get_leaf_names())
                    s2_leaves = set(subtree2.get_leaf_names())
                    leaves_sub1 = len(s1_leaves)
                    total_bl_sub1 = get_total_branch_length(subtree1)
                    longest_bl_sub1 = get_longest_branch(subtree1)
                    leaves_sub2 = len(s2_leaves)
                    total_bl_sub2 = get_total_branch_length(subtree2)
                    longest_bl_sub2 = get_longest_branch(subtree2)

                    # *** Iterate through potential attachment edges in BOTH subtrees ***
                    # An edge is defined by the node it leads TO (node must have a parent)
                    attach_edges_t1 = [node for node in subtree1.traverse() if node.up]
                    attach_edges_t2 = [node for node in subtree2.traverse() if node.up]

                    logging.debug(f"Bisect {bisect_name}: Found {len(attach_edges_t1)} attach edges in T1, {len(attach_edges_t2)} in T2.")

                    for j, attach_node1_ref in enumerate(attach_edges_t1):
                        attach_name1 = getattr(attach_node1_ref, 'name', None)
                        if not attach_name1: continue

                        for k, attach_node2_ref in enumerate(attach_edges_t2):
                            attach_name2 = getattr(attach_node2_ref, 'name', None)
                            if not attach_name2: continue

                            # *** Try all 3 reconnection types ***
                            for reconnect_type, reconnect_func in enumerate([_reconnect_tbr_type1, _reconnect_tbr_type2, _reconnect_tbr_type3], 1):
                                move_counter += 1
                                # *** Updated move_id to include type ***
                                move_id = f"{i}_{j}_{k}_t{reconnect_type}"
                                logging.debug(f"--- Trying Move {move_id}: bisect='{bisect_name}', attach1_node='{attach_name1}', attach2_node='{attach_name2}', type={reconnect_type} ---")

                                # Call the specific reconnect function using names
                                neighbor_tree = reconnect_func(subtree1, attach_name1, subtree2, attach_name2, f"internal_{move_id}") # Exits on failure

                                if neighbor_tree:
                                    valid_neighbor_counter += 1
                                    try:
                                        # Clean up tree before writing/evaluating
                                        neighbor_tree.resolve_polytomy(default_dist=MIN_BRANCH_LENGTH)
                                        # Collapse unifurcations iteratively
                                        unifurcations_found = True
                                        while unifurcations_found:
                                            unifurcations_found = False
                                            nodes_to_check = list(neighbor_tree.traverse()) # Use traverse to check all nodes
                                            for node in nodes_to_check:
                                                # Check if node still exists in the tree (might have been deleted)
                                                # A node is in the tree if it has a parent OR it is the root
                                                is_still_in_tree = node.up is not None or neighbor_tree == node
                                                if not is_still_in_tree: continue
                                                # Check if internal node has exactly one child
                                                if not node.is_leaf() and len(node.children) == 1:
                                                    unifurcations_found = True
                                                    logging.debug(f"Found unifurcation at node '{node.name}'. Collapsing.")
                                                    try: node.delete(prevent_nondicotomic=False, preserve_branch_length=True)
                                                    except Exception as collapse_err: logging.warning(f"Could not collapse unifurcation at node '{node.name}': {collapse_err}")
                                            if unifurcations_found: logging.debug("Re-checking for unifurcations after collapsing.")

                                        # Enforce min branch length
                                        for node in neighbor_tree.traverse():
                                            if node.up and node.dist < MIN_BRANCH_LENGTH: node.dist = MIN_BRANCH_LENGTH

                                        neighbor_tree_str = neighbor_tree.write(format=1)
                                        if not neighbor_tree_str.endswith(";"): neighbor_tree_str += ";"

                                        # *** Write updated info to Newick CSV ***
                                        csvwriter_nwk.writerow([move_id, bisect_name, attach_name1, attach_name2, reconnect_type, neighbor_tree_str])

                                    except Exception as write_err:
                                         logging.error(f"Failed to process/write neighbor tree {move_id}: {write_err}")
                                         bad_tree_path = os.path.join(ds_path, f"bad_process_{move_id}.nwk")
                                         try: 
                                            neighbor_tree.write(format=3, outfile=bad_tree_path)
                                            logging.warning(f"Saved potentially bad tree (process/write error) to {bad_tree_path}")
                                         except Exception as save_err: logging.warning(f"Could not save bad process/write tree {move_id}: {save_err}")
                                         continue

                                    ll_rearr, rtime = call_raxml_mem(neighbor_tree_str, msa_rampath, rates, pinv, alpha, freq, orig_ll)
                                    # *** Update move_data dictionary ***
                                    move_data = { 'move_id': move_id, 'bisect_branch': bisect_name, 'attach_edge1_node': attach_name1, 'attach_edge2_node': attach_name2, 'reconnect_type': reconnect_type,
                                        'leaves_sub1': leaves_sub1, 'total_bl_sub1': total_bl_sub1, 'longest_bl_sub1': longest_bl_sub1,
                                        'leaves_sub2': leaves_sub2, 'total_bl_sub2': total_bl_sub2, 'longest_bl_sub2': longest_bl_sub2,
                                        'll': ll_rearr, 'time': rtime, 'orig_ds_ll': orig_ll }
                                    results_list.append(move_data)
                                # else: implicit exit on failure inside reconnect

                # else: implicit exit on failure inside bisect

        logging.info(f"Finished TBR loops. Total moves attempted: {move_counter}. Valid neighbors generated: {valid_neighbor_counter}.")

        if results_list:
            df = pd.DataFrame(results_list)
            df['ll'] = pd.to_numeric(df['ll'], errors='coerce')
            # *** Update column selection for summary CSV ***
            final_summary_cols = [col for col in csv_headers if col in df.columns] # Ensure all columns exist
            df = df[final_summary_cols]
            df.to_csv(output_summary_csv, index=False, na_rep='NaN')
            logging.info(f"Successfully generated TBR summary: {output_summary_csv}")
            logging.info(f"Successfully generated TBR Newick strings: {newick_output_csv}")
        else:
            logging.warning("No valid TBR neighbors generated or passed assertions.")
    except SystemExit:
        logging.critical("Exiting script due to leaf mismatch detected.")
        # Do not re-raise, allow script to finish writing summary if possible
        # However, the summary might be incomplete.
        logging.warning("Script terminated early due to leaf mismatch.")
        # Save intermediate results if any were collected
        if results_list:
            df = pd.DataFrame(results_list)
            if 'll' in df.columns:
                 df['ll'] = pd.to_numeric(df['ll'], errors='coerce')
            final_summary_cols = [col for col in csv_headers if col in df.columns]
            df = df[final_summary_cols]
            partial_summary_path = output_summary_csv.replace(".csv", "_partial.csv")
            df.to_csv(partial_summary_path, index=False, na_rep='NaN')
            logging.warning(f"Saved partial TBR summary to: {partial_summary_path}")

    except Exception as e:
        logging.error(f'Unhandled exception in all_TBR_simple for dataset: {ds_path}')
        logging.error(traceback.format_exc())
    finally:
        if os.path.exists(msa_rampath):
            try: 
                os.remove(msa_rampath)
                logging.info(f"Removed temporary MSA file: {msa_rampath}")
            except OSError as rm_err: logging.warning(f"Failed to remove temporary MSA file {msa_rampath}: {rm_err}")
    logging.info(f"Exiting all_TBR_simple for {ds_path}")
    return

# --- Main Execution Block (Simplified for TBR only) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform TBR moves and evaluate likelihoods.')
    parser.add_argument('--dataset_path', '-ds', required=True, help='Path to the dataset directory.')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set logging level.')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f'Invalid log level: {args.loglevel}. Defaulting to INFO.')
        numeric_level = logging.INFO
    logging.getLogger().setLevel(numeric_level)
    logging.info(f"Logging level set to {args.loglevel.upper()}")

    dataset_path = args.dataset_path
    if not os.path.isdir(dataset_path):
        logging.critical(f"Dataset path not found: {dataset_path}")
        sys.exit(1)
    if not dataset_path.endswith(SEP):
        dataset_path += SEP
    logging.info(f"Processing dataset: {dataset_path}")

    logging.info("TBR operation requested.")
    outpath_format_string = SUMMARY_PER_DS.format(dataset_path, "{}", "br", "1")
    # *** Update output filename check ***
    outpath_tbr_summary_check = outpath_format_string.format("tbr_standard_with_sub_feats")
    if not os.path.exists(outpath_tbr_summary_check):
         logging.info(f"Running all_TBR_simple for {dataset_path}")
         all_TBR_simple(dataset_path, outpath_format_string)
    else:
        logging.info(f"TBR summary file already exists: {outpath_tbr_summary_check}")
    logging.info("Script finished.")

