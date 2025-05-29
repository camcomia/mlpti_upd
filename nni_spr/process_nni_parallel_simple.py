# process_nni_imported.py
import os
import sys
import glob
import re
# No subprocess needed here
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Import the refactored functions ---
# Assuming the refactored scripts are in the python path or same directory
# Adjust path if necessary, e.g., sys.path.append('/path/to/scripts')
try:
    # Note: Ensure helper functions used by run_spr_analysis are defined within its file
    from run_spr_on_tree import run_spr_analysis
    # Note: Ensure helper functions used by process_spr_results are defined within its file
    from adapted_collect_features import process_spr_results
except ImportError as e:
    print(f"Error importing required functions: {e}")
    print("Make sure the refactored scripts (run_spr_on_tree.py, adapted_collect_features.py) are accessible.")
    sys.exit(1)


def process_single_nni_imported(nni_tree_file, msa_folder_abs, msa_file_abs, stats_file_abs, hybrid_dir_abs):
    """
    Processes a single NNI tree file by calling imported functions.
    """
    base_name = os.path.basename(nni_tree_file)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_prefix = f"[{timestamp}][{base_name}] "

    print(f"{log_prefix}Starting processing...", flush=True)

    # --- Step 1: Run SPR analysis logic ---
    print(f"{log_prefix}Running SPR analysis...", flush=True)
    spr_success, summary_csv_path, newick_csv_path = run_spr_analysis(
        input_tree_path=nni_tree_file,
        msa_path=msa_file_abs,
        params_stats_file=stats_file_abs,
        output_dir=hybrid_dir_abs
    )

    if not spr_success:
        print(f"{log_prefix}SPR analysis failed.", flush=True)
        return nni_tree_file, False, "SPR analysis function failed"
    print(f"{log_prefix}SPR analysis successful.", flush=True)
    print(f"{log_prefix}  Summary file: {summary_csv_path}", flush=True)
    print(f"{log_prefix}  Newick file: {newick_csv_path}", flush=True)


    # --- Step 2: Run Feature Collection logic ---
    # Check if the files were actually created (belt-and-suspenders check)
    if not summary_csv_path or not os.path.isfile(summary_csv_path):
         print(f"{log_prefix}Error: Summary CSV file not found after SPR run: {summary_csv_path}", flush=True)
         return nni_tree_file, False, "Summary CSV missing post-SPR"
    if not newick_csv_path or not os.path.isfile(newick_csv_path):
         print(f"{log_prefix}Error: Newick CSV file not found after SPR run: {newick_csv_path}", flush=True)
         # Don't fail here, adapted_collect_features handles cleanup if summary exists
         # return nni_tree_file, False, "Newick CSV missing post-SPR"
         pass # Allow collect features to proceed and potentially cleanup

    print(f"{log_prefix}Running feature collection...", flush=True)
    collect_success = process_spr_results(
        nni_tree_path=nni_tree_file,
        summary_path=summary_csv_path,
        newick_path=newick_csv_path, # Pass path even if missing, let function handle it
        orig_ds_id=nni_tree_file # Use full path as ID
    )

    if not collect_success:
        print(f"{log_prefix}Feature collection failed.", flush=True)
        return nni_tree_file, False, "Feature collection function failed"
    print(f"{log_prefix}Feature collection successful.", flush=True)

    # --- Success for this file ---
    print(f"{log_prefix}Successfully processed.", flush=True)
    print(f"{log_prefix}{'-'*50}", flush=True)
    return nni_tree_file, True, None

# --- Main Execution Logic (Similar to before) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SPR and Feature Collection by importing functions (Parallel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-ds", "--msa_folder", help="Path to the dataset directory.")
    parser.add_argument("-w", "--max_workers", type=int, default=4, help="Max parallel threads.")

    args = parser.parse_args()
    msa_folder_abs = os.path.abspath(args.msa_folder)
    max_workers = args.max_workers

    start_time = time.time()
    print(f"Starting processing via imports for: {msa_folder_abs}")
    print(f"Using max {max_workers} workers.")

    # Define expected paths
    nni_dir = os.path.join(msa_folder_abs, 'NNI')
    hybrid_dir = os.path.join(msa_folder_abs, 'Hybrid_NNI_SPR')
    msa_file_abs = os.path.join(msa_folder_abs, 'real_msa.phy')
    stats_file_abs = os.path.join(msa_folder_abs, 'real_msa.phy_phyml_stats_bionj.txt')

    # Basic validation
    if not os.path.isdir(nni_dir): print(f"Error: NNI dir not found: {nni_dir}"); sys.exit(1)
    if not os.path.isfile(msa_file_abs): print(f"Error: MSA file not found: {msa_file_abs}"); sys.exit(1)
    if not os.path.isfile(stats_file_abs): print(f"Error: Stats file not found: {stats_file_abs}"); sys.exit(1)

    os.makedirs(hybrid_dir, exist_ok=True)

    # Find NNI Trees
    nni_tree_pattern = os.path.join(nni_dir, 'optimized_*.raxml.bestTree')
    nni_trees = glob.glob(nni_tree_pattern)

    if not nni_trees:
        print(f"No NNI trees found in {nni_dir}. Exiting."); sys.exit(0)

    print(f"Found {len(nni_trees)} NNI trees to process.")

    # --- Parallel Execution ---
    success_count = 0
    failure_count = 0
    tasks_completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tree = {
            executor.submit(process_single_nni_imported, tree, msa_folder_abs, msa_file_abs, stats_file_abs, hybrid_dir): tree
            for tree in nni_trees
        }
        for i, future in enumerate(as_completed(future_to_tree), 1):
            tasks_completed += 1
            tree = future_to_tree[future]
            base_name = os.path.basename(tree)
            print(f"--- Task {tasks_completed}/{len(nni_trees)} completed (for {base_name}) ---", flush=True)
            try:
                origin_file, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    print(f"Task failed for {base_name}. Reported error: {error_msg}", flush=True)
            except Exception as exc:
                failure_count += 1
                print(f'Task for {base_name} generated an exception: {exc}', flush=True)

    # --- Final Summary ---
    duration = time.time() - start_time
    print("\n--- Processing Summary ---", flush=True)
    # ... (Summary print statements as before) ...
    print(f"Total NNI trees found: {len(nni_trees)}", flush=True)
    print(f"Tasks completed: {tasks_completed}", flush=True)
    print(f"Successful tasks: {success_count}", flush=True)
    print(f"Failed tasks: {failure_count}", flush=True)
    print(f"Total execution time: {duration:.2f} seconds", flush=True)


    # --- Final Cleanup ---
    # ... (Cleanup logic as before) ...
    if failure_count == 0 and os.path.exists(nni_dir):
        print(f"\nAll tasks successful. Removing NNI directory: {nni_dir}", flush=True)
        try: shutil.rmtree(nni_dir); print("NNI directory removed.")
        except OSError as e: print(f"Error removing NNI directory {nni_dir}: {e}")
    elif os.path.exists(nni_dir):
        print(f"\nFailures occurred. NOT removing NNI directory: {nni_dir}")

    print("Python script finished.", flush=True)
    sys.exit(1 if failure_count > 0 else 0)