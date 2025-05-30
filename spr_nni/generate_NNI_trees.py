from defs_NNI import *
import glob

def all_nni(newick_file):
	ds_path = newick_file.split("/")
	ds_path = ds_path[:-1]
	ds_path = "/".join(ds_path + [""])

	orig_msa_file = ds_path + MSA_PHYLIP_FILENAME
	stats_filepath = ds_path + PHYML_STATS_FILENAME.format('spr')
	t_orig = get_tree(ds_path, orig_msa_file)
	t_orig.get_tree_root().name = ROOTLIKE_NAME
    
	nwk_str = t_orig.write(format=1)
	tree = Tree(nwk_str, format=1)
	init_recursive_features(tree)
	
	# FEATURES OF STARTING TREE
	start_tree = extract_tree_features(tree)
	
	nni_trees, leaves, bls = perform_nni(tree)
	
	df = extract_nni_features(nni_trees, leaves, bls)
	
	#first, copy msa file to memory and save it:
	msa_rampath = f"/dev/shm/tmp_{random.random()}_{random.random()}_" + ds_path.split(SEP)[-2] #  to be on the safe side (even though other processes shouldn't be able to access it)
	with open(orig_msa_file) as fpr:
		msa_str = fpr.read()
	try:
		with open(msa_rampath, "w") as fpw:
			fpw.write(msa_str)  # don't write the msa string to a variable (or write and release it)
		msa_str = ''

		params_dict = (parse_phyml_stats_output(None, stats_filepath))
		freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"],params_dict["subCT"], params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]
		# df = pd.DataFrame()


		nnis, a, b = perform_nni(tree)
		
		ll_rearrs = []
		n = 1
		for nni in nnis:
			newick_str = nni.write(format=1)
			ll_rearr, rtime = call_raxml_mem(newick_str, msa_rampath, rates, pinv, alpha, freq, n, ds_path)
			# print (ll_rearr, rtime)
			n+=1
			if (rtime == "no ll opt_no time"):
				continue
			ll_rearrs.append(ll_rearr)

	except Exception as e:
		print('could not complete the all_nni function on dataset:', dataset_path, '\nError message:')
		print(e)
		exit()
	finally:
		os.remove(msa_rampath)

		
	optimized_trees = ["{}NNI/optimized_{}.raxml.bestTree".format(ds_path, i+1) for i in range(len(nnis))]
	for t in optimized_trees:
		if (os.path.exists(t)):
			optimize = Tree(t, format=1)
			init_recursive_features(optimize)

	features_after = []
	for i, (opt_tree, bl_info) in enumerate(zip(optimized_trees, bls)):
		if (os.path.exists(opt_tree)):
			opt = Tree(opt_tree, format=1)
			init_recursive_features(opt)
			bl_subtree1_after = opt.search_nodes(name=bl_info[0].name)[0].dist
			bl_subtree2_after = opt.search_nodes(name=bl_info[2].name)[0].dist      

			longest_branch_subtree1_after = opt.search_nodes(name=bl_info[0].name)[0].maxBL #max(node.dist for node in nni_tree.search_nodes(name=bl_info[0].name)[0].traverse())
			longest_branch_subtree2_after = opt.search_nodes(name=bl_info[2].name)[0].maxBL #max(node.dist for node in nni_tree.search_nodes(name=bl_info[2].name)[0].traverse())

			features_after.append({
				"idx": i,
				"S1":bl_info[0].name,
				"S2":bl_info[2].name,
				"bl_S1_a": bl_subtree1_after,
				"bl_S2_a": bl_subtree2_after,
				"maxBL_S1_a": longest_branch_subtree1_after,
				"maxBL_S2_a": longest_branch_subtree2_after,
				"nwk_str": opt.write(format=1)
			})
	features_after = pd.DataFrame(features_after)
	
	df_copy = df.merge(features_after, on=["idx", "S1", "S2"])
	df_copy["ll_rearrs"] = ll_rearrs
	df_copy["ll_orig"] = params_dict["ll"]
	df_copy["ntaxa"] = start_tree["ntaxa"]
	df_copy["tbl"] = start_tree["tbl"]
	df_copy["maxBL"] = start_tree["maxBL"]
	
	df_copy["ll_rearrs"] = df_copy["ll_rearrs"].astype(float)
	df_copy["ll_orig"] = df_copy["ll_orig"].astype(float)
	df_copy["target"] = (df_copy["ll_rearrs"] - df_copy["ll_orig"]) / df_copy["ll_orig"]
	
	df_copy.to_csv('{}dataset.csv'.format(ds_path), index=False)
	directory = f"{ds_path}"
	file_to_keep = "dataset.csv"
	for dirpath, _, filenames in os.walk(directory):
		for filename in filenames:
			if filename != file_to_keep:
				file_path = os.path.join(dirpath, filename)
				os.remove(file_path)
				# print(f"Deleted: {file_path}")

    # if (os.path.exists(f"{ds_path}NNI")):
    #     print (f"Deleting {ds_path}NNI")
    #     shutil.rmtree(f"{ds_path}NNI")
def start_NNI(dataset_path):
        # Executes all processes for a given dataset sequentially.
        print("START NNI:", dataset_path, "at", datetime.datetime.now())
        if os.path.exists(dataset_path):
                all_nni(dataset_path)
        else:
                print ("Path does not exist")
        print("DONE NNI:", dataset_path, "at", datetime.datetime.now())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--starting_tree', '-f', required=True, help='Path to the input tree file (Newick format).')

    args = parser.parse_args()
    start_NNI(args.starting_tree)
###############################################################################

# def extractFeatures(dataset_path):
# 	# Executes all processes for a given dataset sequentially.
# 	print("START extractFeatures:", dataset_path, "at", datetime.datetime.now())
# 	if os.path.exists(dataset_path):
# 		all_nni(dataset_path)
# 	else:
# 		print ("Path does not exist")
# 	print("DONE extractFeatures:", dataset_path, "at", datetime.datetime.now())
	
# if __name__ == "__main__":
# 	starting_trees = glob.glob(f"./test_data/**/real_msa.phy_phyml_tree_spr.txt", recursive=True)
# 	# for s in starting_trees:
# 	# 	extractFeatures(s)
# 	with ProcessPoolExecutor(max_workers=min(6, os.cpu_count()-2)) as executor: #adjust max workers accordingly
# 		executor.map(extractFeatures, starting_trees)

