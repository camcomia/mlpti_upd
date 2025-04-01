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

from defs_PhyAI import *

################################################################################################
########################## begining of 'features extraction' section ###########################
################################################################################################



def get_newick_tree(tree):
	"""
	:param tree: newick tree string or txt file containing one tree
	:return:	tree: a string of the tree in ete3.Tree format
	"""
	if os.path.exists(tree):
		with open(tree, 'r') as tree_fpr:
			tree = tree_fpr.read().strip()
	return tree
	
	
def get_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: list of branch lengths
	"""
	try:
		if type(tree) == str:
			tree = Tree(get_newick_tree(tree), format=1)
		tree_root = tree.get_tree_root()
	except:
		print(tree)
	if len(tree) == 1 and not "(" in tree:    # in one-branch trees, sometimes the newick string is without "(" and ")" so the .iter_decendants returns None
		return [tree.dist]
	branches = []
	for node in tree_root.iter_descendants(): # the root dist is 1.0, we don't want it
		branches.append(node.dist)

	return branches


def get_total_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: total branch lengths
	"""
	branches = get_branch_lengths(tree)
	return sum(branches)


def prune_branch(t_orig, prune_name): # the same function from 'SPR_and_lls' script
	'''
	get (a copy of) both subtrees after pruning
	'''
	t_cp_p = t_orig.copy()  				# the original tree is needed for each iteration
	prune_node_cp = t_cp_p & prune_name     # locate the node in the copied subtree
	assert prune_node_cp.up

	nname = prune_node_cp.up.name
	prune_loc = prune_node_cp
	prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
	t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided
	if not nname:
		nname = "Nnew_p"

	return nname, prune_node_cp, t_cp_p


def dist_between_nodes(t, node1):
	nleaves_between, tbl_between = {},{}
	for node2 in t.get_descendants("levelorder")[::-1]:
		if not node2.name:
			node2.name = "Nnew"
		nname2 = node2.name
		if node1.name == nname2:
			continue

		nleaves_between[nname2] = node1.get_distance(node2, topology_only=True)+1   # +1 to convert between nodes count to edges
		tbl_between[nname2] = node1.get_distance(node2, topology_only=False)

	return nleaves_between, tbl_between


def init_recursive_features(t):
	assert isinstance(t, Tree)
	for node in t.traverse("postorder"):
		if node.is_leaf():
			node.add_feature("cumBL", 0)
			node.add_feature("maxBL", 0)
			node.add_feature("ntaxa", 1)
		else:
			#since it's postorder, we must have already went over its children leaves
			left, right = node.children
			node.add_feature("cumBL", left.cumBL + right.cumBL + left.dist + right.dist)
			node.add_feature("maxBL", max(left.maxBL, right.maxBL, left.dist, right.dist))
			node.add_feature("ntaxa", left.ntaxa + right.ntaxa)


def update_node_features(subtree):
	"""
	:param subtree: a node that needs update. might be None or a leaf
	:return: None
	"""
	left, right = subtree.children
	subtree.cumBL = left.cumBL + right.cumBL + left.dist + right.dist
	subtree.maxBL = max(left.maxBL, right.maxBL, left.dist, right.dist)
	subtree.ntaxa = left.ntaxa + right.ntaxa


def calc_leaves_features(tree_str, move_type):
	t = Tree(newick=tree_str, format=1)
	if move_type != 'prune' and move_type != 'rgft':   # namely move_type in the prune_name to return for the res_tree
		return (t&move_type).dist

	ntaxa = len(t)
	tbl = get_total_branch_lengths(tree_str)

	name2bl, name2pdist_pruned, name2pdist_remaining, name2tbl_pruned, name2tbl_remaining, name2longest_pruned, name2longest_remaining, name2ntaxa, name2ntaxa_pruned, name2ntaxa_remaining, name2pars_pruned, name2parse_remaining = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
	names2bl_dist, names2topology_dist = {}, {}


	nodes_order = []
	first_node = t.get_leaves()[0]
	t.set_outgroup(first_node)

	# We want to traverse the tree by pooling the parent of the remaining tree and moving it to the outgroup -
	# so we do it in a preorder manner
	for node in t.traverse("preorder"):
		nodes_order.append(node)
	nodes_order.pop(0)  # this is t, first_node's nonsense parent

	# initialize tree features in here, and afterwards only rotate tree and update them
	init_recursive_features(t)  # compute features for primary tree

	for node in nodes_order:
		# root the tree with current node's parent
		# when rotating a tree in a preorder manner - the parent's parent (grandparent) becomes the parent's son.
		# so we need to update both of them (if exist)
		nodes_to_update = [node.up]
		while nodes_to_update[-1]:
			nodes_to_update.append(nodes_to_update[-1].up)
		nodes_to_update.pop(-1)  # None
		nodes_to_update.pop(-1)  # the nonesense root

		t.set_outgroup(node)
		for up_node in nodes_to_update[::-1]:
			update_node_features(up_node)

		nname = node.name
		subtree1, subtree2 = t.children
		if nname == '':
			continue

		bl = node.dist * 2 if "Sp" in nname else node.children[1].dist
		name2bl[nname] = bl

		tbl_r = subtree2.cumBL if "Sp" in nname else node.children[1].cumBL  # without the branch that is being pruned
		name2tbl_remaining[nname] = tbl_r
		name2tbl_pruned[nname] = tbl - tbl_r							 	 # with the branch that is being pruned

		longest_p = max(subtree1.maxBL, bl)
		longest_r = subtree2.maxBL if "Sp" in nname else node.children[1].maxBL
		name2longest_pruned[nname] = longest_p
		name2longest_remaining[nname] = longest_r

		ntaxa_p = subtree1.ntaxa if "Sp" in nname else subtree2.ntaxa + 1
		name2ntaxa_pruned[nname] = ntaxa_p
		name2ntaxa_remaining[nname] = ntaxa - ntaxa_p
		##################

		res_dists = dist_between_nodes(t, node) if move_type == "prune" else ["unrelevant", "unrelevant"]
		names2topology_dist[nname] = res_dists[0]  # res_dists is a dict
		names2bl_dist[nname] = res_dists[1]  # res_dists is a dict


	d = OrderedDict([("bl", name2bl), ("longest", max(longest_p, longest_r)),
					 ("ntaxa_p", name2ntaxa_pruned), ("ntaxa_r", name2ntaxa_remaining),
					 ("tbl_p", name2tbl_pruned), ("tbl_r", name2tbl_remaining),
					 ("top_dist", names2topology_dist), ("bl_dist", names2bl_dist),
					 ("longest_p", name2longest_pruned), ("longest_r", name2longest_remaining)])

	return d



################################################################################################
############################# end of 'features extraction' section #############################
################################################################################################
################################################################################################
##################### begining of 'organizing features in csv' section #########################
################################################################################################


def index_additional_rgft_features(df_rgft, ind, prune_name, rgft_name, res_bl, features_dict_prune):
	df_rgft.loc[ind, FEATURES["top_dist"]] = features_dict_prune['top_dist'][prune_name][rgft_name]
	df_rgft.loc[ind, FEATURES["bl_dist"]] = features_dict_prune['bl_dist'][prune_name][rgft_name]
	df_rgft.loc[ind, FEATURES["res_bl"]] = res_bl

	return df_rgft


def index_shared_features(dff, ind, edge, move_type, features_dicts_dict):
	d_ll = format(dff.loc[ind, "ll"] - dff.loc[ind, "orig_ds_ll"], '.20f')
	dff.loc[ind, LABEL.format(move_type)] = d_ll  # LABEL

	#*tbl of orig tree will be calculated via 'RandomForest_learning' script		#f1  (FeaturePruning)
	dff.loc[ind, FEATURES["bl"]] = features_dicts_dict["bl"][edge] 					#f2
	dff.loc[ind, FEATURES["longest"]] = features_dicts_dict["longest"]				#f3


	#index subtrees features
	for subtype in ["p", "r"]:
		dff.loc[ind, FEATURES["ntaxa_{}".format(subtype)]] = features_dicts_dict["ntaxa_{}".format(subtype)][edge]  		#f4,5
		dff.loc[ind, FEATURES["tbl_{}".format(subtype)]] = features_dicts_dict["tbl_{}".format(subtype)][edge]  			#f6,7
		dff.loc[ind, FEATURES["longest_{}".format(subtype)]] = features_dicts_dict["longest_{}".format(subtype)][edge]	#f8,9

	return dff


def collect_features(ds_path, step_number, outpath_prune, outpath_rgft, tree_type='bionj'):
	dfr = pd.read_csv(TREES_PER_DS.format(ds_path, step_number), index_col=0)
	df_prune = pd.read_csv(outpath_prune, index_col=0)
	df_rgft = pd.read_csv(outpath_rgft, index_col=0)

	features_prune_dicts_dict = calc_leaves_features(ds_path + PHYML_TREE_FILENAME.format(tree_type), "prune")

	for i, row in dfr.iterrows():
		ind = row.name
		print(ind)
		tree = row["newick"]
		if row["rgft_name"] == SUBTREE2:	# namely the remaining subtree
			features_rgft_dicts_dict = calc_leaves_features(tree, "rgft")  # msa will be truncated within the function
		if not "subtree" in row["rgft_name"]:
			res_bl = calc_leaves_features(tree, row["prune_name"])   # res tree is before bl optimization
			df_prune = index_shared_features(df_prune, ind, row["prune_name"], "prune",  features_prune_dicts_dict)
			df_rgft = index_shared_features(df_rgft, ind, row["rgft_name"], "rgft", features_rgft_dicts_dict)
			df_rgft = index_additional_rgft_features(df_rgft, ind, row["prune_name"], row["rgft_name"], res_bl, features_prune_dicts_dict)   # also prune dict because for 2 features i didn't want to comp dict within each rgft iteration (needed to compute on the starting tree)

	df_prune.to_csv(outpath_prune)  # runover existing one (with lls only) to fill in all features
	df_rgft.to_csv(outpath_rgft)    # runover existing one (with lls only) to fill in all features
	
	return

################################################################################################
######################### end of 'organizing features in csv' section ##########################
################################################################################################



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='perform all SPR moves')
	parser.add_argument('--dataset_path', '-ds', required=True)
	args = parser.parse_args()

	dataset_path = args.dataset_path
	step_number = "1"
	outpath_prune = SUMMARY_PER_DS.format(dataset_path, "prune", "br", step_number)
	outpath_rgft = SUMMARY_PER_DS.format(dataset_path, "rgft", "br", step_number)
	
	collect_features(dataset_path, step_number, outpath_prune, outpath_rgft)