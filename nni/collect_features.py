from defs_NNI import *

def all_nni(newick_file):
    """
    Performs all NNI rearrangements on a given starting tree, extracts features before and after 
    rearrangements, evaluates likelihood changes using RAxML, and outputs results to a CSV file.
    """
    # Derive dataset path
    ds_path = os.path.join(*newick_file.split("/")[:-1]) + "/"
    orig_msa_file = os.path.join(ds_path, MSA_PHYLIP_FILENAME)
    stats_filepath = os.path.join(ds_path, PHYML_STATS_FILENAME.format('bionj'))

    # Load and initialize starting tree
    t_orig = get_tree(ds_path, orig_msa_file)
    t_orig.get_tree_root().name = ROOTLIKE_NAME
    tree = Tree(t_orig.write(format=1), format=1)
    init_recursive_features(tree)

    start_tree = extract_tree_features(tree)
    nni_trees, leaves, bls = perform_nni(tree)
    df = extract_nni_features(nni_trees, leaves, bls)

    # Use shared memory for temporary MSA storage
    msa_rampath = f"/dev/shm/tmp_{os.path.basename(ds_path.strip('/'))}"
    try:
        with open(orig_msa_file) as fpr:
            msa_str = fpr.read()
        with open(msa_rampath, "w") as fpw:
            fpw.write(msa_str)
        msa_str = ''  # Clear from memory

        # Parse phyML stats
        params = parse_phyml_stats_output(None, stats_filepath)
        freq = [params["fA"], params["fC"], params["fG"], params["fT"]]
        rates = [params["subAC"], params["subAG"], params["subAT"], 
                 params["subCG"], params["subCT"], params["subGT"]]
        pinv = params["pInv"]
        alpha = params["gamma"]

        ll_rearrs = []
        for i, nni in enumerate(nni_trees, 1):
            nwk_str = nni.write(format=1)
            ll_rearr, rtime = call_raxml_mem(nwk_str, msa_rampath, rates, pinv, alpha, freq, i, ds_path)
            if rtime != "no ll opt_no time":
                ll_rearrs.append(ll_rearr)

    except Exception as e:
        print(f"[ERROR] Could not complete all_nni for dataset: {ds_path}\n{e}")
        exit()
    finally:
        if os.path.exists(msa_rampath):
            os.remove(msa_rampath)

    # Extract post-optimization branch length features
    features_after = []
    optimized_trees = [f"{ds_path}NNI/optimized_{i+1}.raxml.bestTree" for i in range(len(nni_trees))]
    for i, (opt_tree_path, bl_info) in enumerate(zip(optimized_trees, bls)):
        if os.path.exists(opt_tree_path):
            opt_tree = Tree(opt_tree_path, format=1)
            init_recursive_features(opt_tree)

            node1 = opt_tree.search_nodes(name=bl_info[0].name)[0]
            node2 = opt_tree.search_nodes(name=bl_info[2].name)[0]

            features_after.append({
                "idx": i,
                "nwk_str": opt_tree.write(format=1),
                "S1": node1.name,
                "S2": node2.name,
                "bl_S1_a": node1.dist,
                "bl_S2_a": node2.dist,
                "maxBL_S1_a": node1.maxBL,
                "maxBL_S2_a": node2.maxBL,
            })

    # Combine pre- and post-optimization features
    df_post = pd.DataFrame(features_after)
    df_merged = df.merge(df_post, on=["idx", "S1", "S2"])
    df_merged["ll_rearrs"] = ll_rearrs
    df_merged["ll_orig"] = float(params["ll"])
    df_merged["ntaxa"] = start_tree["ntaxa"]
    df_merged["tbl"] = start_tree["tbl"]
    df_merged["maxBL"] = start_tree["maxBL"]

    # Define target variable: normalized likelihood improvement
    df_merged["ll_rearrs"] = df_merged["ll_rearrs"].astype(float)
    df_merged["ll_orig"] = df_merged["ll_orig"].astype(float)
    df_merged["target"] = (df_merged["ll_rearrs"] - df_merged["ll_orig"]) / df_merged["ll_orig"]

    # Save final dataset
    df_merged.to_csv(os.path.join(ds_path, "dataset.csv"), index=False)

    # Clean up temporary tree output directory
    nni_output_dir = os.path.join(ds_path, "NNI")
    if os.path.exists(nni_output_dir):
        shutil.rmtree(nni_output_dir)

def extractFeatures(dataset_path):
    """
    Executes the NNI pipeline on a given dataset.
    """
    start_time = datetime.datetime.now()
    print(f"[{start_time}] STARTED: extractFeatures on {dataset_path}")

    if os.path.exists(dataset_path):
        all_nni(dataset_path)
    else:
        print(f"[{datetime.datetime.now()}] ERROR: Path does not exist - {dataset_path}")

    end_time = datetime.datetime.now()
    print(f"[{end_time}] COMPLETED: extractFeatures on {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform all NNI moves and extract features for each dataset.')
    parser.add_argument('--training_folder', '-tf', required=True, help='Path to the training folder containing subdirectories with MSA and tree files.')
    args = parser.parse_args()

    # Collect all valid starting tree paths
    starting_trees = glob.glob(os.path.join(args.training_folder, "*/real_msa.phy_phyml_tree_bionj.txt"))
    
    if not starting_trees:
        print(f"[{datetime.datetime.now()}] No starting trees found in {args.training_folder}. Exiting.")
        exit()

    print(f"[{datetime.datetime.now()}] Found {len(starting_trees)} datasets. Starting parallel processing...")

    # Use parallel processing to extract features
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() - 2)) as executor:
        executor.map(extractFeatures, starting_trees)

    print(f"[{datetime.datetime.now()}] All datasets processed.")


