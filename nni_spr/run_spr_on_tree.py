#########################################################################
##                 Copyright (C). All Rights Reserved.                   ##
##      "Harnessing machine learning to guide                            ##
##                              phylogenetic-tree search algorithms"     ##
##                                                                       ##
## by Dana Azouri, Shiran Abadi, Yishay Mansour, Itay Mayrose, Tal Pupko ##
##                                                                       ##
## Adapted script to run SPR generation and likelihood calculation       ##
## on a specific input tree (e.g., an NNI neighbor).  
#########################################################################

# Standard library imports
import os
import sys
import re
import shutil
import argparse
import random
import csv
import pandas as pd
from subprocess import Popen, PIPE, STDOUT
from io import StringIO

# Required external libraries (Script will fail here if not installed)
from ete3 import PhyloTree, Tree
from Bio import AlignIO

# from defs_PhyAI import * # Uncomment if defs_PhyAI is preferred source of constants

# --- Configuration ---
RAXML_NG_SCRIPT = "raxml-ng" # Adjust this path if needed

# --- Constants (defined locally for clarity/safety) ---
PHYLIP_FORMAT = "phylip-relaxed"
ROOTLIKE_NAME = "ROOT_LIKE" # Name assigned to root if unnamed
SEP = "/"
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"


# --- Helper Functions---

def get_msa_from_file(msa_file_path):
    """Reads MSA from a phylip file."""
    if not os.path.exists(msa_file_path):
        print(f"Error: MSA file not found at {msa_file_path}")
        return None
    try:
        msa = AlignIO.read(msa_file_path, PHYLIP_FORMAT)
        return msa
    except Exception as e:
        print(f"Error reading MSA file {msa_file_path}: {e}")
        return None

def parse_phyml_stats_output(stats_filepath):
    """
    Parses a PhyML stats file to extract model parameters.
    Returns: dictionary with parameters or None on error.
    """
    res_dict = dict.fromkeys(["ll", "pInv", "gamma",
                              "fA", "fC", "fG", "fT",
                              "subAC", "subAG", "subAT", "subCG", "subCT", "subGT"], None)
    res_dict["path"] = stats_filepath
    if not os.path.exists(stats_filepath):
        print(f"Error: Stats file not found at {stats_filepath}")
        return None
    try:
        with open(stats_filepath) as fpr: content = fpr.read()
        ll_match = re.search(r"Log-likelihood:\s+([-\d.]+)", content)
        # Ensure 'll' key exists even if parsing fails
        res_dict["ll"] = ll_match.group(1).strip() if ll_match else None
        gamma_match = re.search(r"Gamma shape parameter:\s+([\d.]+)", content)
        pinv_match = re.search(r"Proportion of invariant.*?:\s+([\d.]+)", content)
        if gamma_match: res_dict['gamma'] = gamma_match.group(1).strip()
        if pinv_match: res_dict['pInv'] = pinv_match.group(1).strip()
        freq_matches = re.findall(r"f\(([ACGT])\)\s*=\s*([\d.]+)", content)
        for nuc, freq in freq_matches: res_dict["f" + nuc] = freq
        sub_matches = re.findall(r"([ACGT])\s*<->\s*([ACGT])\s*([\d.]+)", content)
        for nuc1, nuc2, rate in sub_matches:
            pair = "".join(sorted((nuc1, nuc2)))
            res_dict["sub" + pair] = rate
        required_params = ["pInv", "gamma", "fA", "fC", "fG", "fT",
                           "subAC", "subAG", "subAT", "subCG", "subCT", "subGT", "ll"] # Check ll too
        missing = [p for p in required_params if res_dict.get(p) is None]
        if missing:
            print(f"Warning: Could not parse parameters {missing} from {stats_filepath}")
            # Only fail if critical params (not just ll) are missing
            if any(p != 'll' for p in missing):
                 print("Error: Critical model parameters missing. Cannot proceed.")
                 return None
    except Exception as e:
        print(f"Error parsing stats file {stats_filepath}: {e}")
        return None
    return res_dict

def parse_raxmlNG_content(content):
    """
    Parses RAxML-NG output content. Extracts ll and time.
    Includes commented-out code for parsing other model parameters.
    Returns: dictionary with all keys, but only ll/time actively parsed.
    """
    print(f"RAxML-NG output: {content}\n")
    res_dict = dict.fromkeys(["ll", "pInv", "gamma", "fA", "fC", "fG", "fT",
                              "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                              "time"], "")
    ll_re = re.search(r"Final LogLikelihood:\s+([-\d.]+)", content)
    if ll_re:
        res_dict["ll"] = ll_re.group(1).strip()
    elif re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content):
        ll_ini = re.search(r"initial LogLikelihood:\s+([-\d.]+)", content)
        if ll_ini:
            print("Warning: RAxML-NG optimization issue, using initial LogLikelihood.")
            res_dict["ll"] = ll_ini.group(1).strip()
        else:
             res_dict["ll"] = 'unknown raxml-ng error, check "parse_raxmlNG_content" function'
             print(f"Error: Could not parse RAxML-NG likelihood. Content snippet:\n{content[:500]}")
    else:
        ll_ini = re.search(r"initial LogLikelihood:\s+([-\d.]+)", content)
        if ll_ini:
             print("Warning: RAxML-NG Final LogLikelihood not found, using initial.")
             res_dict["ll"] = ll_ini.group(1).strip()
        else:
             res_dict["ll"] = 'unknown raxml-ng error, check "parse_raxmlNG_content" function'
             print(f"Error: Could not parse RAxML-NG likelihood. Content snippet:\n{content[:500]}")

    # --- Time Parsing (Active) ---
    rtime = re.search(r"Elapsed time:\s+([\d.]+)\s+seconds", content)
    if rtime:
        res_dict["time"] = rtime.group(1).strip()
    else:
        res_dict["time"] = 'no ll opt_no time'

    if 'unknown raxml-ng error' in str(res_dict["ll"]): pass
    elif res_dict["ll"] == "":
        print("Critical Error: RAxML-NG likelihood calculation failed or could not be parsed.")
        res_dict["ll"] = None
    try:
        res_dict["ll"] = float(res_dict["ll"])
    except (ValueError, TypeError): pass
    return res_dict

def call_raxml_mem(tree_str, msa_tmpfile, model_params, is_input_tree=False):
    """
    Calls RAxML-NG to evaluate likelihood. Returns (likelihood, time) tuple or just likelihood.
    """
    required_keys = ["subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
                     "pInv", "gamma", "fA", "fC", "fG", "fT"]
    if not all(k in model_params and model_params[k] is not None for k in required_keys):
        print("Error: Missing model parameters for RAxML-NG call.")
        fail_val = (None, 'Parameter Error')
        return fail_val[0] if is_input_tree else fail_val
    try:
        rates_str = "/".join(map(str, [model_params["subAC"], model_params["subAG"], model_params["subAT"],
                                       model_params["subCG"], model_params["subCT"], model_params["subGT"]]))
        pinv_str = str(model_params["pInv"])
        alpha_str = str(model_params["gamma"])
        freq_str = "/".join(map(str, [model_params["fA"], model_params["fC"], model_params["fG"], model_params["fT"]]))
        model_line_params = f'GTR{{{rates_str}}}+I{{{pinv_str}}}+G{{{alpha_str}}}+F{{{freq_str}}}'
    except KeyError as e:
        print(f"Error: Missing parameter {e}.")
        fail_val = (None, 'Parameter Error')
        return fail_val[0] if is_input_tree else fail_val
    except Exception as e:
        print(f"Error formatting model string: {e}")
        fail_val = (None, 'Formatting Error')
        return fail_val[0] if is_input_tree else fail_val

    tree_rampath = f"/dev/shm/raxml_temp_tree_{random.random()}_{random.random()}.nwk"
    raxml_output = ""
    res_dict = {"ll": None, "time": "Exec Error"}
    try:
        with open(tree_rampath, "w") as fpw: fpw.write(tree_str)
        cmd = [RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile, '--threads', '2',
               '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params,
               '--nofiles', '--tree', tree_rampath, '--precision', '8', '--blopt', 'nr_safe']
        p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=STDOUT, text=True)
        raxml_stdout, _ = p.communicate()
        raxml_output = raxml_stdout
        res_dict = parse_raxmlNG_content(raxml_output)
        if res_dict['ll'] is None: pass
    except FileNotFoundError:
        print(f"Error: RAxML-NG executable not found at '{RAXML_NG_SCRIPT}'.")
        res_dict = {"ll": None, "time": "RAxML Not Found"}
    except Exception as e:
        print(f"Error during RAxML-NG execution for {tree_rampath}: {e}")
        print(f"RAxML Output was:\n{raxml_output}")
        res_dict = {"ll": None, "time": "Exec Error"}
    finally:
        if os.path.exists(tree_rampath):
            try: os.remove(tree_rampath)
            except OSError as e: print(f"Warning: Could not remove temporary file {tree_rampath}: {e}")
    return res_dict['ll'] if is_input_tree else (res_dict['ll'], res_dict['time'])

def prune_branch(t_orig, prune_node_name):
    """Prunes a branch, returns (parent_name, pruned_subtree, remaining_tree)."""
    try:
        t_cp_p = t_orig.copy()
        prune_node_cp = t_cp_p.search_nodes(name=prune_node_name)
        if not prune_node_cp: return None, None, None
        prune_node_cp = prune_node_cp[0]
        if prune_node_cp.is_root(): return None, None, None
        parent_node = prune_node_cp.up
        if not parent_node: return None, None, None
        parent_name = parent_node.name
        pruned_subtree = prune_node_cp.detach()
        parent_node.delete(preserve_branch_length=True)
        return parent_name, pruned_subtree, t_cp_p
    except Exception as e:
        print(f"Error during prune operation for node '{prune_node_name}': {e}")
        return None, None, None

def regraft_branch(remaining_tree, pruned_subtree, rgft_node_name, original_parent_name):
    """
    Regrafts the pruned_subtree onto the remaining_tree.
    Returns tuple: (regrafted_tree, preserve_flag)
    """
    preserve = False
    try:
        t_curr = remaining_tree.copy()
        rgft_node_cp = t_curr.search_nodes(name=rgft_node_name)
        if not rgft_node_cp: return None, preserve
        rgft_node_cp = rgft_node_cp[0]
        if rgft_node_cp.is_root(): return None, preserve
        rgft_parent = rgft_node_cp.up
        if not rgft_parent: return None, preserve
        original_rgft_dist = rgft_node_cp.dist
        new_branch_length = original_rgft_dist / 2.0 if original_rgft_dist > 0 else 0.0
        new_internal_node = Tree()
        new_internal_node.name = original_parent_name
        rgft_node_cp.detach()
        new_internal_node.add_child(rgft_node_cp, dist=new_branch_length)
        new_internal_node.add_child(pruned_subtree.copy(), dist=new_branch_length)
        rgft_parent.add_child(new_internal_node, dist=new_branch_length)
        if original_parent_name == ROOTLIKE_NAME: preserve = True
        return t_curr, preserve
    except Exception as e:
        print(f"Error during regraft operation onto node '{rgft_node_name}': {e}")
        return None, preserve

def generate_spr_neighbors_and_calc_likelihoods(input_tree, input_tree_ll, msa_rampath, model_params, newick_csv_writer):
    """
    Generates SPR neighbors, calculates their likelihoods, and writes Newick strings
    (including intermediate subtrees).
    """
    spr_likelihood_results = []
    t_orig = input_tree.copy()

    needs_naming = False
    for node in t_orig.traverse():
        if not node.is_leaf() and not node.name: needs_naming = True; break
    if needs_naming:
        print("Info: Input tree lacks internal node names. Assigning temporary names (N#).")
        i = 0
        for node in t_orig.traverse():
            if not node.is_leaf() and not node.name: node.name = f"N{i}"; i += 1

    possible_prune_nodes = [n for n in t_orig.traverse() if not n.is_root()]
    if not possible_prune_nodes: return []

    print(f"Generating and evaluating SPR neighbors for {len(possible_prune_nodes)} prune locations...")
    for i, prune_node in enumerate(possible_prune_nodes):
        prune_name = prune_node.name
        if not prune_name: continue

        original_parent_name, pruned_subtree, remaining_tree = prune_branch(t_orig, prune_name)
        if not remaining_tree or not pruned_subtree: continue

        try:
            pruned_nwk = pruned_subtree.write(format=1)
            remaining_nwk = remaining_tree.write(format=1)
            newick_csv_writer.writerow([f"{i},0", prune_name, SUBTREE1, pruned_nwk])
            newick_csv_writer.writerow([f"{i},1", prune_name, SUBTREE2, remaining_nwk])
        except Exception as e:
            print(f"Error writing intermediate subtrees for prune node {prune_name}: {e}")
            continue

        for j, rgft_node in enumerate(remaining_tree.iter_descendants()):
            rgft_name = rgft_node.name
            if not rgft_name: continue
            if original_parent_name == rgft_name: continue

            regrafted_tree, preserve = regraft_branch(remaining_tree, pruned_subtree, rgft_name, original_parent_name)

            if regrafted_tree:
                try:
                    neighbor_tree_str = regrafted_tree.write(format=1, format_root_node=preserve)
                except Exception as e: continue

                index_str = f"{i},{j+2}"
                try:
                    newick_csv_writer.writerow([index_str, prune_name, rgft_name, neighbor_tree_str])
                except Exception as e: print(f"Error writing Newick row to CSV: {e}")

                ll_rearr, rtime = call_raxml_mem(neighbor_tree_str, msa_rampath, model_params)

                if ll_rearr is not None:
                    spr_likelihood_results.append({
                        "prune_name": prune_name, "rgft_name": rgft_name,
                        "ll": ll_rearr, "time": rtime,
                    })
                else:
                     print(f"  Skipping neighbor (prune={prune_name}, rgft={rgft_name}) due to RAxML error (LL is None).")

    print(f"Finished generating SPR neighbors. Total successful likelihood calculations: {len(spr_likelihood_results)}")
    return spr_likelihood_results

# --- Main Execution Logic ---

# --- Core Logic Function (Refactored) ---
def run_spr_analysis(input_tree_path, msa_path, params_stats_file, output_dir):
    """
    Main function to run SPR generation and likelihood calculation.
    Saves summary CSV and Newick CSV.

    Args:
        input_tree_path (str): Path to the input tree file (e.g., NNI neighbor).
        msa_path (str): Path to the multiple sequence alignment file.
        params_stats_file (str): Path to the PhyML stats file (for model params & orig LL).
        output_dir (str): Directory to save the output CSV files.

    Returns:
        tuple: (success_flag (bool), summary_csv_path (str/None), newick_csv_path (str/None))
               Paths are returned upon success, None otherwise.
    """
    print(f"Starting SPR analysis for tree: {input_tree_path}", flush=True)
    # --- Define output paths ---
    input_tree_filename = os.path.basename(input_tree_path)
    # Extract index reliably, handling potential variations if needed
    match = re.search(r'optimized_(\d+)', input_tree_filename)
    if not match:
         print(f"Error: Could not extract index from input tree filename: {input_tree_filename}")
         return False, None, None
    index_str = match.group(1)
    output_base = os.path.join(output_dir, index_str) # Use index for base name

    summary_csv_path = f"{output_base}.spr_summary.csv"
    newick_csv_path = f"{output_base}.spr_newicks.csv"
    print(f"Output summary CSV: {summary_csv_path}", flush=True)

    # --- Create Output Directory ---
    try:
        # output_dir comes from args now, create it if needed
        if not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             print(f"Created output directory: {output_dir}", flush=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}", flush=True)
        return False, None, None

    # --- Read Input Tree ---
    try:
        input_tree = Tree(input_tree_path, format=1)
        if not input_tree.name:
            input_tree.name = ROOTLIKE_NAME
        input_tree_str = input_tree.write(format=1)
        print(f"Successfully read input tree: {input_tree_path}", flush=True)
    except Exception as e:
        print(f"Error reading input tree file {input_tree_path}: {e}", flush=True)
        return False, None, None

    # --- Read MSA ---
    msa = get_msa_from_file(msa_path)
    if not msa:
        print(f"Error: No MSA read from {msa_path}.")
        return False, None, None

    # --- Copy MSA to /dev/shm ---
    # Create a unique ID based on inputs to avoid collisions in parallel runs
    unique_id = abs(hash(input_tree_path + msa_path + params_stats_file))
    msa_rampath = f"/dev/shm/msa_{unique_id}.phy"
    try:
        # Use copyfile for potentially better performance than copyfileobj
        shutil.copyfile(msa_path, msa_rampath)
    except Exception as e:
        print(f"Warning: Error copying MSA to {msa_rampath}: {e}. Using original path.", flush=True)
        msa_rampath = msa_path # Fallback to original path

    # --- Read Model Parameters ---
    model_params = parse_phyml_stats_output(params_stats_file)
    if not model_params:
        print("Failed to parse model parameters. Cannot proceed.", flush=True)
        if msa_rampath != msa_path and os.path.exists(msa_rampath):
            try: os.remove(msa_rampath)
            except OSError: pass
        return False, None, None

    # Check if orig_ds_ll ('ll' key in params) is valid
    orig_ds_ll_str = model_params.get('ll')
    try:
        orig_ds_ll = float(orig_ds_ll_str)
    except (ValueError, TypeError, AttributeError):
        print(f"Error: Invalid or missing 'll' (orig_ds_ll) in params file: {params_stats_file}. Cannot proceed.", flush=True)
        if msa_rampath != msa_path and os.path.exists(msa_rampath):
            try: os.remove(msa_rampath)
            except OSError: pass
        return False, None, None

    # --- Calculate Likelihood of Input NNI Tree ---
    nni_tree_ll = call_raxml_mem(input_tree_str, msa_rampath, model_params, is_input_tree=True)
    if nni_tree_ll is None or not isinstance(nni_tree_ll, (int, float)): # Check if LL calculation failed or returned error string
        print(f"Failed to calculate likelihood for the NNI input tree (Result: {nni_tree_ll}). Cannot proceed.", flush=True)
        if msa_rampath != msa_path and os.path.exists(msa_rampath):
            try: os.remove(msa_rampath)
            except OSError: pass
        return False, None, None

    # --- Generate SPR Neighbors, Calculate Likelihoods, and Write Newicks ---
    results = []
    fp_newick = None
    spr_success_flag = True # Assume success unless exception occurs
    try:
        fp_newick = open(newick_csv_path, "w", newline='')
        csvwriter = csv.writer(fp_newick)
        csvwriter.writerow(['index', 'prune_name', 'rgft_name', 'newick'])
        results = generate_spr_neighbors_and_calc_likelihoods(input_tree, nni_tree_ll, msa_rampath, model_params, csvwriter)
    except Exception as e:
        print(f"An error occurred during SPR generation/calculation: {e}", flush=True)
        spr_success_flag = False # Mark as failure
    finally:
        if fp_newick: fp_newick.close()
        if msa_rampath != msa_path and os.path.exists(msa_rampath):
            try: 
                os.remove(msa_rampath)
            except OSError as e: 
                print(f"Warning: Could not remove temporary MSA {msa_rampath}: {e}", flush=True)

    # --- Save Likelihood Summary Results ---
    save_success_flag = False
    if results and spr_success_flag: # Only save if SPR step didn't fail and we have results
        try:
            results_df = pd.DataFrame(results)
            results_df["nni_tree_ll"] = nni_tree_ll
            results_df["orig_ds_ll"] = orig_ds_ll
            results_df['ll'] = pd.to_numeric(results_df['ll'], errors='coerce')
            results_df.to_csv(summary_csv_path, index=False)
            print(f"Successfully saved SPR likelihood summary to: {summary_csv_path}", flush=True)
            save_success_flag = True
        except Exception as e:
            print(f"Error saving summary results to CSV {summary_csv_path}: {e}", flush=True)
            save_success_flag = False
    elif not results and spr_success_flag:
        print("SPR generation completed, but no likelihood results were generated (check RAxML calls).", flush=True)
        save_success_flag = True # No data to save, but process didn't crash
    else:
        print("SPR generation failed or no results produced.", flush=True)
        save_success_flag = False # Mark failure if SPR step failed

    print(f"SPR analysis finished for tree: {input_tree_path}", flush=True)
    # Return success only if both SPR generation and saving worked (or if no results needed saving)
    overall_success = spr_success_flag and save_success_flag
    return overall_success, summary_csv_path if overall_success else None, newick_csv_path if overall_success else None

# --- Keep the __main__ block for command-line compatibility ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SPR neighbors for a given input tree...')
    parser.add_argument('--input_tree', '-t', required=True, help='Path to the input tree file (Newick format).')
    parser.add_argument('--msa', '-m', required=True, help='Path to the multiple sequence alignment file (Phylip format).')
    parser.add_argument('--params_stats_file', '-p', required=True, help='Path to the file containing model parameters AND original dataset LL.')
    parser.add_argument('--output_dir', '-o', default=None, help='Directory to save output CSV files. Defaults to input tree directory.')

    args = parser.parse_args()
    print(f"Starting run_on_tree.py on {args.input_tree}...")

    # Determine output directory if not provided
    output_dir_main = args.output_dir
    if output_dir_main is None:
        output_dir_main = os.path.dirname(args.input_tree)
        if not output_dir_main: output_dir_main = "." # Use current dir if path is just filename

    # Basic path validation before calling the main function
    if not os.path.isfile(args.input_tree): 
        print(f"Error: Input tree file not found: {args.input_tree}")
        sys.exit(1)
    if not os.path.isfile(args.msa): 
        print(f"Error: MSA file not found: {args.msa}")
        sys.exit(1)
    if not os.path.isfile(args.params_stats_file): 
        print(f"Error: Parameter stats file not found: {args.params_stats_file}")
        sys.exit(1)

    print(f"Starting run_spr_analysis on {args.input_tree}...")
    success, _, _ = run_spr_analysis(args.input_tree, args.msa, args.params_stats_file, output_dir_main)

    # Exit with appropriate status code
    sys.exit(0 if success else 1)