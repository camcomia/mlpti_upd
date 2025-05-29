import math
import os, re, shutil, argparse, random
from Bio import AlignIO
from ete3 import *  
from subprocess import Popen, PIPE, STDOUT
import datetime
import pandas as pd

#########################################################################################################
# Constants
RAXML_NG_SCRIPT = "raxml-ng" # path to the raxml-ng executable
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
N_ESTIMATORS = 70
MSA_PHYLIP_FILENAME = "real_msa.phy"
PHYML_STATS_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_stats_{0}.txt"
PHYML_TREE_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_tree_{0}.txt"
NNI_TREES_FILENAME = "nni_trees.csv"
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

def parse_raxmlNG_content(content):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ll"], "")

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

	return res_dict


def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq, n, ds_path, df):
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
									 pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
									 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree_str)
		p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '1', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--prefix', f'{ds_path}/nni_{n}', '--tree', tree_rampath, '--redo', 'blopt', 'nr_safe', 'precision', '8'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()
		extra_files = [
			f'{ds_path}nni_{n}.raxml.rba',
			f'{ds_path}nni_{n}.raxml.log',
			f'{ds_path}nni_{n}.raxml.startTree',
			f'{ds_path}nni_{n}.raxml.bestModel',
			f'{ds_path}nni_{n}.raxml.bestTreeCollapsed',
            f'{ds_path}nni_{n}.raxml.ckp',
        	f'{ds_path}nni_{n}.raxml.reduced.phy',
		]
		
		for file in extra_files:
			if os.path.exists(file):
				os.remove(file)
		
		res_dict = parse_raxmlNG_content(raxml_output)
		ll = res_dict['ll']

		# Writing the newick tree and corresponding likelihood to the dataframe then removing file generated by raxml to optimize space
		newick_tree = ''
		if os.path.exists(f'{ds_path}nni_{n}.raxml.bestTree'):
			with open(f'{ds_path}nni_{n}.raxml.bestTree', "r") as fpr:
				newick_tree = fpr.read().strip()
		
			if newick_tree:
				new_row = pd.DataFrame({"newick": [newick_tree], "ll": [ll]})
				df = pd.concat([df, new_row], ignore_index=True)
				os.remove(f'{ds_path}nni_{n}.raxml.bestTree')

			
	except Exception as e:
		print(e)
		exit()
	finally:
		os.remove(tree_rampath)

	return df

########################################################################################################################

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
        for node in tree.traverse("postorder"):
            if not node.is_leaf() and not node.is_root():
                nnis = _generate_nni_for_node(tree, node)
                nni_trees.extend(nnis)
        return nni_trees
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
    children = node.children
    
    if len(children) != 2:
        raise ValueError("NNI requires bifurcating trees.")
    
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

    tree1_child1.detach()
    tree1_sibling.detach()

    # Reattach them in the swapped configuration
    tree1_node.add_child(tree1_sibling)
    tree1_parent.add_child(tree1_child1)
    
    nni_trees.append(tree1)
  
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

    tree2_child2.detach()
    tree2_sibling.detach()

    # Reattach them in the swapped configuration
    tree2_node.add_child(tree2_sibling)
    tree2_parent.add_child(tree2_child2)
    
    nni_trees.append(tree2)

    return nni_trees

def all_nni(newick_file):
	ds_path = newick_file.split("/")
	ds_path = ds_path[:-1]
	ds_path = "/".join(ds_path + [""])

	orig_msa_file = ds_path + MSA_PHYLIP_FILENAME
	stats_filepath = ds_path + PHYML_STATS_FILENAME.format('bionj')
	t_orig = get_tree(ds_path, orig_msa_file)
	t_orig.get_tree_root().name = ROOTLIKE_NAME
    
	nwk_str = t_orig.write(format=1)
	tree = Tree(nwk_str, format=1)
	
	df = pd.DataFrame(columns=['newick', 'll'])
	folder = os.path.dirname(newick_file)
	nni_file = os.path.join(folder, NNI_TREES_FILENAME)
	
	#first, copy msa file to memory and save it:
	msa_rampath = "/scratch/ctcomia/tmp" + ds_path.split(SEP)[-2] #  to be on the safe side (even though other processes shouldn't be able to access it)
	with open(orig_msa_file) as fpr:
		msa_str = fpr.read()
	try:
		with open(msa_rampath, "w") as fpw:
			fpw.write(msa_str)  # don't write the msa string to a variable (or write and release it)
		msa_str = ''

		params_dict = (parse_phyml_stats_output(None, stats_filepath))
		freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"],params_dict["subCT"], params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

		nnis = perform_nni(tree)
		print(f'Finished perform nni and generated {len(nnis)} NNI neighbors')
		n = 1
		for nni in nnis:
			newick_str = nni.write(format=1)
			df = call_raxml_mem(newick_str, msa_rampath, rates, pinv, alpha, freq, n, ds_path, df)
			n+=1

		print("Finished calculating ll for all NNI neighbors. Retrieving top 20% trees for SPR step")
		top20_df = get_top_percent_trees(df)
		try:
			top20_df.to_csv(nni_file.replace(".csv", "_top20.csv"), index=False)
			print("Finished writing to file:", nni_file)
		except Exception as e:
			print("could not complete writing trees to file:", nni_file, ". writing all trees instead")
			df.to_csv(nni_file.replace(".csv", "_failed_full_df.csv"), index=False) # full list of nni trees
            

	except Exception as e:
		print('could not complete the all_nni function on dataset:', newick_file, '\nError message:')
		print(e)
		exit()
	finally:
		os.remove(msa_rampath)

def get_top_percent_trees(df):
	df['ll'] = pd.to_numeric(df['ll'], errors='coerce')
	initial_rows = len(df)
	# Drop rows where 'll' could not be converted or was originally NaN
	df.dropna(subset=['ll'], inplace=True)
	# Count rows after dropping NaNs
	valid_rows = len(df)
	if initial_rows > valid_rows:
		print(f"Warning: Dropped {initial_rows - valid_rows} rows due to non-numeric or missing 'll' values.")
	if valid_rows == 0:
		print("Error: No valid rows with numeric 'll' values found.")
	else:
		df_sorted = df.sort_values(by='ll', ascending=False)
		num_top_rows = math.floor(valid_rows * 0.2) # Calculate the number of rows for the top 20%
		print(f"{valid_rows} valid rows will be sorted by likelihood and the top {num_top_rows} rows will be selected for SPR step")
		return df_sorted.head(num_top_rows)
	
def start_NNI(msa_file):
	if not os.path.exists(msa_file):
		print("File not found:", msa_file)
		return
	print("START NNI:", msa_file, "at", datetime.datetime.now())
	all_nni(msa_file)
	print("DONE NNI:", msa_file, "at", datetime.datetime.now())
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NNI neighbors for a given input tree, calculate their likelihoods using RAxML-NG, and save results mimicking SPR_and_lls.py output.')
    parser.add_argument('--starting_tree', '-f', required=True, help='Path to the input tree file (Newick format).')

    args = parser.parse_args()
    start_NNI(args.starting_tree)
    print("Finished NNI for all nodes for", args.starting_tree)