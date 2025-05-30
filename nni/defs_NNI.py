import os, re, shutil, argparse, random
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from collections import OrderedDict
from io import StringIO
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
from Bio import Phylo
from ete3 import *
import csv
from subprocess import Popen, PIPE, STDOUT
import datetime, subprocess, argparse, os
from concurrent.futures import ProcessPoolExecutor
import glob


RAXML_NG_SCRIPT = "./raxml-ng_v0.9.0/raxml-ng"
#########################################################################################################
SEP = "/"
MODEL_DEFAULT = "GTR+I+G"
OPT_TYPE = "br"
PHYLIP_FORMAT = "phylip-relaxed"
REARRANGEMENTS_NAME = "rearrangement"
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"
ROOTLIKE_NAME = "ROOT_LIKE"
GROUP_ID = 'group_id'
KFOLD = 10
N_ESTIMATORS = 100


MSA_PHYLIP_FILENAME = "real_msa.phy"
PHYML_STATS_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_stats_{0}.txt"
PHYML_TREE_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_tree_{0}.txt"
SUMMARY_PER_DS = "{}ds_summary_{}_{}_step{}.csv"
TREES_PER_DS = "{}newicks_step{}.csv"
LEARNING_DATA = "learning_{}_step{}.csv"
DATA_WITH_PREDS = "with_preds_merged_{}.csv"
SCORES_PER_DS = "scores_per_ds_{}.csv"
#########################################################################################################
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


def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq, n, ds_path):
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
									 pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
									 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree_str)
		os.makedirs(f'{ds_path}NNI/', exist_ok=True)
		p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '2', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--prefix', f'{ds_path}/NNI/optimized_{n}', '--tree', tree_rampath, '--redo', "--precision", "8", "--blopt", "nr_safe"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		# print (" ".join([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '2', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--prefix', f'{ds_path}/NNI/optimized_{n}', '--tree', tree_rampath, '--redo', "--precision" "8" "--blopt" "nr_safe"]))
		# p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '4', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--nofiles', '--tree', tree_rampath], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()
		
		# extra_files = [
		# 	f'{ds_path}/NNI/optimized_{n}.raxml.rba',
		# 	f'{ds_path}/NNI/optimized_{n}.raxml.log',
		# 	f'{ds_path}/NNI/optimized_{n}.raxml.startTree',
		# 	f'{ds_path}/NNI/optimized_{n}.raxml.bestModel'
		# ]
		# for file in extra_files:
		# 	if os.path.exists(file):
		# 		os.remove(file)
		# print (raxml_output)
        # Define the folder path and file extensions to keep
		folder_path = f"{ds_path}/NNI"
		keep_extensions = {'.bestTree'}  # Set of file extensions to keep

		# Iterate over files in the folder
		for filename in os.listdir(folder_path):
			file_path = os.path.join(folder_path, filename)

			# Skip directories
			if os.path.isdir(file_path):
				continue

			# Check if the file extension is not in the keep_extensions set
			if not any(filename.endswith(ext) for ext in keep_extensions):
				os.remove(file_path)
				# print (file_path)
				
		res_dict = parse_raxmlNG_content(raxml_output)
		ll = res_dict['ll']
		rtime = res_dict['time']

	except Exception as e:
		print(msa_tmpfile.split(SEP)[-1][3:])
		print(e)
		exit()
	finally:
		os.remove(tree_rampath)

	return ll, rtime

########################################################################################################################

def extract_nni_features(nni_trees, nni_leaves, nni_branch_lengths):
    """
    Extracts features from the resulting NNI trees.
    """
    features_list = []
    for i, (nni_tree, leaves_info, bl_info) in enumerate(zip(nni_trees, nni_leaves, nni_branch_lengths)):
        init_recursive_features(nni_tree)

        subtree1_leaves = leaves_info[1]
        subtree2_leaves = leaves_info[3]

        bl_subtree1_before = bl_info[1]
        bl_subtree2_before = bl_info[3]

        longest_branch_subtree1_before = bl_info[0].maxBL #max(node.dist for node in bl_info[0].traverse())
        longest_branch_subtree2_before = bl_info[2].maxBL #max(node.dist for node in bl_info[2].traverse())

        cumBL_subtree1_before = nni_tree.search_nodes(name=bl_info[0].name)[0].cumBL #max(node.dist for node in bl_info[0].traverse())
        cumBL_subtree2_before = nni_tree.search_nodes(name=bl_info[2].name)[0].cumBL #max(node.dist for node in bl_info[0].traverse())

        features_list.append({
                "idx": i,
                "S1":bl_info[0].name,
                "S2":bl_info[2].name,
                "ntaxa_S1": subtree1_leaves,
                "ntaxa_S2": subtree2_leaves,
                "bl_S1_b": bl_subtree1_before,
                "bl_S2_b": bl_subtree2_before,
                "maxBL_S1_b": longest_branch_subtree1_before,
                "maxBL_S2_b": longest_branch_subtree2_before,
                "cumBL_S1": cumBL_subtree1_before,
                "cumBL_S2": cumBL_subtree2_before
        })
    
    return pd.DataFrame(features_list)

def extract_tree_features(tree):
    """
    Extracts features from the given phylogenetic tree.
    """
    assert isinstance(tree, Tree)
    init_recursive_features(tree)
    
    num_leaves = len(tree.get_leaves())
    total_branch_length = sum(node.dist for node in tree.traverse())
    longest_branch = max(node.dist for node in tree.traverse())
    
    return {
        "ntaxa": num_leaves,
        "tbl": total_branch_length,
        "maxBL": longest_branch,
    }

def init_recursive_features(t):
    assert isinstance(t, Tree)
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node.add_feature("cumBL", 0)
            node.add_feature("maxBL", 0)
            node.add_feature("ntaxa", 1)
        else:
            #since it's postorder, we must have already went over its children leaves
            if (len(node.children) == 2):
                left, right = node.children
                node.add_feature("cumBL", left.cumBL + right.cumBL + left.dist + right.dist)
                node.add_feature("maxBL", max(left.maxBL, right.maxBL, left.dist, right.dist))
                node.add_feature("ntaxa", left.ntaxa + right.ntaxa)
            elif (len(node.children) > 2):
                cumBL = sum([(n.cumBL + n.dist) for n in node.children])
                maxBL = max([n.maxBL for n in node.children] + [n.dist for n in node.children])
                ntaxa = sum([n.ntaxa for n in node.children])
                node.add_feature("cumBL", cumBL)
                node.add_feature("maxBL", maxBL)
                node.add_feature("ntaxa", ntaxa)

def perform_nni(tree, edge_to_break=None):
    """
    Perform Nearest Neighbor Interchange (NNI) on a phylogenetic tree.
    
    Args:
        tree (ete3.Tree): Input phylogenetic tree.
        edge_to_break (tuple): Edge to break for NNI, represented as (parent, child).
                               If None, all possible NNI rearrangements are generated.
    
    Returns:
        list: List of trees resulting from NNI rearrangements.
    """
    if edge_to_break is None:
        # Generate all possible NNI rearrangements
        nni_trees = []
        nni_leaves = []
        nni_branch_lengths = []
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                nnis, leaves, bls = _generate_nni_for_node(tree, node)
                nni_trees.extend(nnis)
                nni_leaves.extend(leaves)
                nni_branch_lengths.extend(bls)
        return nni_trees, nni_leaves, nni_branch_lengths
    else:
        # Perform NNI on the specified edge
        parent, child = edge_to_break
        node = tree.search_nodes(name=child)[0]
        return _generate_nni_for_node(tree, node)

def _generate_nni_for_node(tree, node):
    """
    Generate NNI rearrangements for a given internal node.
    
    Args:
        node (ete3.TreeNode): Internal node to perform NNI on.
    
    Returns:
        list: List of trees resulting from NNI rearrangements.
    """
    if node.is_leaf() or node.is_root():
        raise ValueError("NNI can only be performed on internal, non-root nodes.")
    
    # Get the parent and children of the node
    parent = node.up
    children = node.children
    
    if len(children) != 2:
        raise ValueError("NNI requires bifurcating trees.")
    
    # Get the sibling of the node
    sibling = [child for child in parent.children if child != node][0]
    
    # Perform the two possible NNI swaps
    nni_trees = []
    
    # Swap 1: Swap one child with the sibling
    tree1 = tree.copy("newick")  # Create a deep copy of the tree
    
    tree1_node = tree1.search_nodes(name=node.name)
    if not tree1_node:
        raise ValueError(f"Node {node.name} not found in the copied tree.")
    tree1_node = tree1_node[0]
    
    tree1_parent = tree1_node.up
    tree1_sibling = [child for child in tree1_parent.children if child != tree1_node][0]
    
    # Detach one child and the sibling
    tree1_child1 = tree1_node.children[0]

    #NUMBER OF LEAVES OF SIBLING1
    tree1_child1_num_leaves = len(tree1_child1.get_leaves()) if not tree1_child1.is_leaf() else 1
    tree1_sibling_num_leaves = len(tree1_sibling.get_leaves()) if not tree1_sibling.is_leaf() else 1
    tree1_child1.detach()
    tree1_sibling.detach()

    leaves_rem_tree = len(tree1.get_leaves())
    
    # Reattach them in the swapped configuration
    tree1_node.add_child(tree1_sibling)
    tree1_parent.add_child(tree1_child1)
    
    nni_trees.append(tree1)
    
    # Calculate total branch length
    total_length_child = sum([node.dist for node in tree1_child1.traverse()])
    total_length_sibling = sum([node.dist for node in tree1_sibling.traverse()])
  
    # Swap 2: Swap the other child with the sibling
    tree2 = tree.copy("newick")  # Create a deep copy of the tree
    tree2_node = tree2.search_nodes(name=node.name)
    if not tree2_node:
        raise ValueError(f"Node {node.name} not found in the copied tree.")
    tree2_node = tree2_node[0]
    tree2_parent = tree2_node.up
    tree2_sibling = [child for child in tree2_parent.children if child != tree2_node][0]
    
    # Detach the other child and the sibling
    tree2_child2 = tree2_node.children[1]

    #NUMBER OF LEAVES OF SIBLING2
    tree2_child2_num_leaves = len(tree2_child2.get_leaves()) if not tree2_child2.is_leaf() else 1

    tree2_child2.detach()
    tree2_sibling.detach()

    leaves_rem_tree = len(tree2.get_leaves())
    
    # Reattach them in the swapped configuration
    tree2_node.add_child(tree2_sibling)
    tree2_parent.add_child(tree2_child2)
    
    nni_trees.append(tree2)
    
    # Calculate total branch length
    total_length_child = sum([node.dist for node in tree2_child2.traverse()])
    total_length_sibling = sum([node.dist for node in tree2_sibling.traverse()])
    
    leaves_nni1 = (tree1_child1.name, tree1_child1_num_leaves, tree1_sibling.name, tree1_sibling_num_leaves)
    leaves_nni2 = (tree2_child2.name, tree2_child2_num_leaves, tree2_sibling.name, tree1_sibling_num_leaves)
    leaves_nni = (leaves_nni1, leaves_nni2)
    bl_nni1 = (tree1_child1, tree1_child1.dist, tree1_sibling, tree1_sibling.dist)
    bl_nni2 = (tree2_child2, tree2_child2.dist, tree2_sibling, tree2_sibling.dist)
    bl_nni = (bl_nni1, bl_nni2)

    return nni_trees, leaves_nni, bl_nni

