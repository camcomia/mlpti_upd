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
RAXML_NG_SCRIPT = "raxml-ng"

################################################################################################
############### begining of 'parsing rearrangements and PhyML outputs' section #################
################################################################################################

def get_msa_from_file(msa_file_path):
	#open file if exists
	if not os.path.exists(msa_file_path):
		return None
	try:
		msa = AlignIO.read(msa_file_path, PHYLIP_FORMAT)
	except:
		return None
	return msa


def get_msa_properties(msa):
	"""
	:param msa: bio.AlignIO format or path to msa file
	:return:
	"""
	if isinstance(msa, str):
		msa = get_msa_from_file(msa)
	ntaxa = len(msa)
	nchars = msa.get_alignment_length()

	return ntaxa, nchars


def get_newick_tree(tree):
	"""
	:param tree: newick tree string or txt file containing one tree
	:return:	tree: a string of the tree in ete3.Tree format
	"""
	if os.path.exists(tree):
		with open(tree, 'r') as tree_fpr:
			tree = tree_fpr.read().strip()
	return tree


def cp_internal_names(treepath_no_internal, treepath_with_internal):
	with open(treepath_with_internal) as fp:
		with_names = fp.read()
	with open(treepath_no_internal) as fp:
		nonames = fp.read()

	with_names_bls = re.findall(":(\d*\.?\d+)", with_names)
	nonames_bls = re.findall(":(\d*\.?\d+)", nonames)
	while len(set(with_names_bls)) != len(with_names_bls):
		u = [k for (k, v) in Counter(with_names_bls).items() if v > 1][0]
		ix = with_names_bls.index(u)
		with_names_bls[ix] = u + "1"
		with_names = with_names.replace(u, u + "1", 1)


	dict = {with_names_bls[i]: nonames_bls[i] for i in range(len(with_names_bls))}

	regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
	try:
		new_str = regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], with_names)
	except:
		print(treepath_no_internal)

	with open(treepath_no_internal, 'r') as fp:
		tree_str = fp.read()
		if re.search("\)N", tree_str):
			return
	with open(treepath_no_internal, 'w') as fp:
		fp.write(new_str)
		
		
def parse_phyml_stats_output(msa_filepath, stats_filepath):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ntaxa", "nchars", "ll",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "pInv", "gamma",
							  "path"], "")
	
	if msa_filepath:
		res_dict['ntaxa'], res_dict['nchars'] = (str(x) for x in get_msa_properties(get_msa_from_file(msa_filepath)))
	
	res_dict["path"] = stats_filepath
	try:
		with open(stats_filepath) as fpr:
			content = fpr.read()
		
		# likelihood
		res_dict["ll"] = re.search("Log-likelihood:\s+(.*)", content).group(1).strip()
		
		# gamma (alpha parameter) and proportion of invariant sites
		gamma_regex = re.search("Gamma shape parameter:\s+(.*)", content)
		pinv_regex = re.search("Proportion of invariant:\s+(.*)", content)
		if gamma_regex:
			res_dict['gamma'] = gamma_regex.group(1).strip()
		if pinv_regex:
			res_dict['pInv'] = pinv_regex.group(1).strip()
		
		# Nucleotides frequencies
		for nuc in "ACGT":
			nuc_freq = re.search("  - f\(" + nuc + "\)\= (.*)", content).group(1).strip()
			res_dict["f" + nuc] = nuc_freq
		
		# substitution frequencies
		for nuc1 in "ACGT":
			for nuc2 in "ACGT":
				if nuc1 < nuc2:
					nuc_freq = re.search(nuc1 + " <-> " + nuc2 + "(.*)", content).group(1).strip()
					res_dict["sub" + nuc1 + nuc2] = nuc_freq
	except:
		print("Error with:", res_dict["path"], res_dict["ntaxa"], res_dict["nchars"])
		return
	return res_dict


def return_ll(tree_dirpath, msa_file, filename, br_mode):
	stats_filepath = SEP.join([tree_dirpath, "{}_phyml_{}_{}.txt".format(filename, "stats", br_mode)])
	try:
		res_dict = parse_phyml_stats_output(msa_file, stats_filepath)
		ll_rearr = float(res_dict["ll"])
	except:
		ll_rearr = None
		pass
		#print("does not exist or empty")

	return ll_rearr



################################################################################################
################## end of 'parsing rearrangements and PhyML outputs' section ###################
################################################################################################
################################################################################################
######################### begining of 'generate SPR neigbors' section ##########################
################################################################################################

def prune_branch(t_orig, prune_name):
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

	return nname, prune_node_cp, t_cp_p


def regraft_branch(t_cp_p, rgft_node, prune_node_cp, rgft_name, nname, preserve=False):
	'''
	get a tree with the 2 concatenated subtrees
	'''

	new_branch_length = rgft_node.dist /2
	t_temp = PhyloTree()  			   # for concatenation of both subtrees ahead, to avoid polytomy
	t_temp.add_child(prune_node_cp)
	t_curr = t_cp_p.copy()
	rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree

	rgft_loc = rgft_node_cp.up
	rgft_node_cp.detach()
	t_temp.add_child(rgft_node_cp, dist=new_branch_length)
	t_temp.name = nname
	rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting
	if nname == "ROOT_LIKE":  # (4)
		t_temp.delete()
		preserve = True  # preserve the name of the root node, as this is a REAL node in this case

	return t_curr, preserve


def add_internal_names(tree_file, tree_file_cp_no_internal, t_orig):
	shutil.copy(tree_file, tree_file_cp_no_internal)
	for i, node in enumerate(t_orig.traverse()):
		if not node.is_leaf():
			node.name = "N{}".format(i)
	t_orig.write(format=3, outfile=tree_file)   # runover the orig file with no internal nodes names


def get_tree(ds_path, msa_file):
	tree_file = ds_path + PHYML_TREE_FILENAME.format("bionj")
	tree_file_cp_no_internal = ds_path + PHYML_TREE_FILENAME.format("bionj_no_internal")
	if not os.path.exists(tree_file_cp_no_internal):
		t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=1)
		add_internal_names(tree_file, tree_file_cp_no_internal, t_orig)
	else:
		t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=3)

	return t_orig


def save_rearr_file(trees_dirpath, rearrtree, filename, runover=False):
	if not os.path.exists(trees_dirpath):
		os.makedirs(trees_dirpath)
	tree_path = trees_dirpath + filename + ".txt"
	if runover or not os.path.exists(tree_path):
		rearrtree.write(format=1, outfile=tree_path)

	return tree_path


def parse_raxmlNG_content(content):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ll", "pInv", "gamma",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "time"], "")

	# likelihood
	ll_re = re.search("Final LogLikelihood:\s+(.*)", content)
	if ll_re:
		res_dict["ll"] = ll_re.group(1).strip()
	elif re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content):   # temp, till next version is available
		ll_ini = re.search("initial LogLikelihood:\s+(.*)", content)
		if ll_ini:
			res_dict["ll"] = ll_ini.group(1).strip()
	else:
		res_dict["ll"] = 'unknown raxml-ng error, check "parse_raxmlNG_content" function'


	# gamma (alpha parameter) and proportion of invariant sites
	gamma_regex = re.search("alpha:\s+(\d+\.?\d*)\s+", content)
	pinv_regex = re.search("P-inv.*:\s+(\d+\.?\d*)", content)
	if gamma_regex:
		res_dict['gamma'] = gamma_regex.group(1).strip()
	if pinv_regex:
		res_dict['pInv'] = pinv_regex.group(1).strip()

	# Nucleotides frequencies
	nucs_freq = re.search("Base frequencies.*?:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if nucs_freq:
		for i,nuc in enumerate("ACGT"):
			res_dict["f" + nuc] = nucs_freq.group(i+1).strip()

	# substitution frequencies
	subs_freq = re.search("Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
	if subs_freq:
		for i,nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):  # todo: make sure order
			res_dict["sub" + nuc_pair] = subs_freq.group(i+1).strip()

	# Elapsed time of raxml-ng optimization
	rtime = re.search("Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
	if rtime:
		res_dict["time"] = rtime.group(1).strip()
	else:
		res_dict["time"] = 'no ll opt_no time'

	return res_dict


def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq):
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
									 pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
									 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree_str)

		p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '2', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--nofiles', '--tree', tree_rampath], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()

		res_dict = parse_raxmlNG_content(raxml_output)
		####FOR RAXML LOGS####
		#print("raxml content: ", raxml_output)
		######################
		ll = res_dict['ll']
		rtime = res_dict['time']

	except Exception as e:
		print(msa_tmpfile.split(SEP)[-1][3:])
		print(e)
		exit()
	finally:
		os.remove(tree_rampath)

	return ll, rtime


def all_SPR(ds_path, outpath):
	orig_msa_file = ds_path + MSA_PHYLIP_FILENAME
	stats_filepath = ds_path + PHYML_STATS_FILENAME.format('bionj')
	t_orig = get_tree(ds_path, orig_msa_file)
	t_orig.get_tree_root().name = ROOTLIKE_NAME

	OUTPUT_TREES_FILE = TREES_PER_DS.format(ds_path, '1')
	with open(OUTPUT_TREES_FILE, "w", newline='') as fpw:
		csvwriter = csv.writer(fpw)
		csvwriter.writerow(['', 'prune_name', 'rgft_name', 'newick'])

	# first, copy msa file to memory and save it:
	msa_rampath = "/dev/shm/tmp" + ds_path.split(SEP)[-2] #  to be on the safe side (even though other processes shouldn't be able to access it)
	with open(orig_msa_file) as fpr:
		msa_str = fpr.read()
	try:
		with open(msa_rampath, "w") as fpw:
			fpw.write(msa_str)  # don't write the msa string to a variable (or write and release it)
		msa_str = ''

		params_dict = (parse_phyml_stats_output(None, stats_filepath))
		freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"],params_dict["subCT"], params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]
		df = pd.DataFrame()
		
		############## generate all SPRs and copute it likelihoods #############
		## (1) avoid moves to a branch in the pruned subtree
		## (2) avoid moves to the branch of a sibiling (retaining the same topology)
		## (3) avoid moves to the branch leading to the parent of the (retaining the same topology)
		## (4) handle the automatic "ROOTLIKE" node of ETE trees -
		##     if the PRUNE location is around the rootlike --> delete the "ROOT_LIKE" subnode when pasting in dest,
		##     and preserve the name of the REAL rootnode when converting back to newick
		########################################################################
		for i, prune_node in enumerate(t_orig.iter_descendants("levelorder")):
			prune_name = prune_node.name
			nname, subtree1, subtree2 = prune_branch(t_orig, prune_name) # subtree1 is the pruned subtree. subtree2 is the remaining subtree
			with open(OUTPUT_TREES_FILE, "a", newline='') as fpa:
				csvwriter = csv.writer(fpa)
				csvwriter.writerow([str(i)+",0", prune_name, SUBTREE1, subtree1.write(format=1)])
				csvwriter.writerow([str(i)+",1", prune_name, SUBTREE2, subtree2.write(format=1)])

			for j, rgft_node in enumerate(subtree2.iter_descendants("levelorder")): # traversing over subtree2 capture cases (1) and (3)
				ind = str(i) + "," + str(j)
				rgft_name = rgft_node.name
				if nname == rgft_name: # captures case (2)
					continue
				rearr_tree, preserve = regraft_branch(subtree2, rgft_node, subtree1, rgft_name, nname)
				if preserve:  # namely, (4) is True
					neighbor_tree_str = rearr_tree.write(format=1, format_root_node=True)
				else:
					neighbor_tree_str = rearr_tree.write(format=1)
					
				### save tree to file by using "append"
				with open(OUTPUT_TREES_FILE, "a", newline='') as fpa:
					csvwriter = csv.writer(fpa)
					csvwriter.writerow([ind, prune_name, rgft_name, neighbor_tree_str])

				ll_rearr, rtime = call_raxml_mem(neighbor_tree_str, msa_rampath, rates, pinv, alpha, freq)

				df.loc[ind, "prune_name"], df.loc[ind, "rgft_name"] = prune_name, rgft_name
				df.loc[ind, "prune_name"], df.loc[ind, "rgft_name"] = prune_name, rgft_name
				df.loc[ind, "time"] = rtime
				df.loc[ind, "ll"] = ll_rearr

		df["orig_ds_ll"] = float(params_dict["ll"])
		df.to_csv(outpath.format("prune"))
		df.to_csv(outpath.format("rgft"))

	except Exception as e:
		print('could not complete the all_SPR function on dataset:', dataset_path, '\nError message:')
		print(e)
		exit()
	finally:
		os.remove(msa_rampath)
	return


################################################################################################
########################### end of 'generate SPR neigbors' section #############################
################################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='perform all SPR moves')
	parser.add_argument('--dataset_path', '-ds', required=True)
	args = parser.parse_args()
	
	dataset_path = args.dataset_path
	# original method calls:
	# outpath_prune = SUMMARY_PER_DS.format(dataset_path, "prune", "br", "")
	# outpath_rgft = SUMMARY_PER_DS.format(dataset_path, "rgft", "br", step_number="1")

	outpath_prune = SUMMARY_PER_DS.format(dataset_path, "prune", "br", "1")
	outpath_rgft = SUMMARY_PER_DS.format(dataset_path, "rgft", "br", "1")
	
	if not os.path.exists(outpath_rgft) or not os.path.exists(outpath_prune):
		all_SPR(dataset_path, SUMMARY_PER_DS.format(dataset_path, "{}", "br", "1"))
