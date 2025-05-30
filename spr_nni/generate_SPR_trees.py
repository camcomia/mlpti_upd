from defs_PhyAI import *

RAXML_NG_SCRIPT = "./raxml-ng_v0.9.0/raxml-ng"
PHYML_SCRIPT = "./PhyML-3.1/PhyML-3.1_linux64"
################################################################
def prune_branch(t_orig, prune_name):
    """
    get (a copy of) both subtrees after pruning
    """
    t_cp_p = t_orig.copy()  				# the original tree is needed for each iteration
    prune_node_cp = t_cp_p & prune_name     # locate the node in the copied subtree
    assert prune_node_cp.up

    nname = prune_node_cp.up.name
    prune_loc = prune_node_cp
    prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
    t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

    return nname, prune_node_cp, t_cp_p


def regraft_branch(t_cp_p, rgft_node, prune_node_cp, rgft_name, nname, preserve=False):
    """
    get a tree with the 2 concatenated subtrees
    """

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

def get_tree(ds_path, msa_file):
    tree_file = ds_path + "real_msa.phy_phyml_tree_bionj.txt"
    tree_file_cp_no_internal = ds_path + "real_msa.phy_phyml_tree_bionj_no_internal.txt"
    if not os.path.exists(tree_file_cp_no_internal):
        t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=1)
        add_internal_names(tree_file, tree_file_cp_no_internal, t_orig)
    else:
        t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=3)

    return t_orig

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

################################################################

def generate_neighbors(ds_path):
    orig_msa_file = ds_path + "real_msa.phy"
    stats_filepath = ds_path + PHYML_STATS_FILENAME.format('bionj')
    t_orig = get_tree(ds_path, orig_msa_file)
    t_orig.get_tree_root().name = "ROOT_LIKE"

    #SPR MOVES
    spr_neighbors = []
    clear_directory(ds_path + "SPR_neighbors")

    msa_rampath = "/dev/shm/tmp" + ds_path.split(SEP)[-2] #  to be on the safe side (even though other processes shouldn't be able to access it)
    with open(orig_msa_file) as fpr:
        msa_str = fpr.read()
    try:
        with open(msa_rampath, "w") as fpw:
            fpw.write(msa_str)  # don't write the msa string to a variable (or write and release it)
        msa_str = ''

        params_dict = (parse_phyml_stats_output(None, stats_filepath))
        freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"],params_dict["subCT"], params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]
       
        n = 1
        for i, prune_node in enumerate(t_orig.iter_descendants("levelorder")):
            prune_name = prune_node.name
            nname, subtree1, subtree2 = prune_branch(t_orig, prune_name) # subtree1 is the pruned subtree. subtree2 is the remaining subtree
            
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
                
                save_spr(rearr_tree, preserve, ds_path, n)
                # call_raxml_mem(neighbor_tree_str, msa_rampath, rates, pinv, alpha, freq, ds_path, n)
                # print (n)
                n += 1
    except Exception as e:
        print('could not complete the all_SPR function on dataset:', ds_path, '\nError message:')
        print(e)
        exit()
    finally:
        if (os.path.exists(msa_rampath)):
            os.remove(msa_rampath)
    return

def clear_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove dir and all contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.mkdir(path)

def save_spr(rearr_tree, preserve, ds_path, n):
    try:
        os.mkdir(f'{ds_path}SPR_neighbors/{n}')
        outfile=f'{ds_path}SPR_neighbors/{n}'
        # print (outfile)
        if preserve:  # namely, (4) is True
            rearr_tree.write(format=1, format_root_node=True, outfile=f'{ds_path}SPR_neighbors/{n}/spr.startTree')
        else:
            rearr_tree.write(format=1, outfile=f'{ds_path}SPR_neighbors/{n}/spr.startTree')
       
        msa_file = ds_path + "real_msa.phy"
        shutil.copy(msa_file, f'{ds_path}SPR_neighbors/{n}')
        msa_file = f'{ds_path}SPR_neighbors/{n}/real_msa.phy'
       
        try:
            q = Popen([PHYML_SCRIPT,"-i", msa_file, "-u", f'{ds_path}SPR_neighbors/{n}/spr.startTree', "-m", "012345","-f", "m", "-v", "e", "-a", "e", "-c", "4", "-o", "l", "-d", "nt", "-n", "1", "-b", "0", "--no_memory_check", "--run_id", "spr"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            q.communicate()[0]
        except Exception as e:
            print (e)
                
        
        folder_path = f"{ds_path}SPR_neighbors/{n}"
        keep_extensions = {'.startTree', '.phy', '.txt'}  # Set of file extensions to keep

        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Check if the file extension is not in the keep_extensions set
            if not any(filename.endswith(ext) for ext in keep_extensions):
                os.remove(file_path)

    except Exception as e:
        print(e)
        exit()

    return
        
def call_raxml_mem(tree_str, msa_tmpfile, rates, pinv, alpha, freq, ds_path, n):
    model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
                                        pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
                                        freq="{{{0}}}".format("/".join(freq)))

    # create tree file in memory and not in the storage:
    tree_rampath = "/dev/shm/" + str(random.random())  + str(random.random()) + "tree"  # the var is the str: tmp{dir_suffix}

    try:
        with open(tree_rampath, "w") as fpw:
            fpw.write(tree_str)
            
        os.mkdir(f'{ds_path}/SPR_neighbors/{n}')
        p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_tmpfile,'--threads', '2', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--prefix', f'{ds_path}/SPR_neighbors/{n}/spr', '--tree', tree_rampath, '--redo', "--precision", "8", "--blopt", "nr_safe"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        p.communicate()[0]
        
        msa_file = ds_path + "real_msa.phy"
        shutil.copy(msa_file, f'{ds_path}SPR_neighbors/{n}')
        # print (msa_file, f'{ds_path}SPR_neighbors/{n}')
        msa_file = f'{ds_path}SPR_neighbors/{n}/real_msa.phy'
        try:
            # print (" ".join([PHYML_SCRIPT,"-i", msa_file, "-u", f'{ds_path}SPR_neighbors/{n}/spr.raxml.bestTree', "-m", "012345","-f", "m", "-v", "e", "-a", "e", "-c", "4", "-o", "l", "-d", "nt", "-n", "1", "-b", "0", "--no_memory_check", "--run_id", "spr"]))
            q = Popen([PHYML_SCRIPT,"-i", msa_file, "-u", f'{ds_path}SPR_neighbors/{n}/spr.raxml.bestTree', "-m", "012345","-f", "m", "-v", "e", "-a", "e", "-c", "4", "-o", "l", "-d", "nt", "-n", "1", "-b", "0", "--no_memory_check", "--run_id", "spr"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
            q.communicate()[0]
        except Exception as e:
            print (e)
                
        
        folder_path = f"{ds_path}/SPR_neighbors/{n}"
        keep_extensions = {'.bestTree', '.phy', '.txt'}  # Set of file extensions to keep

        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            # Check if the file extension is not in the keep_extensions set
            if not any(filename.endswith(ext) for ext in keep_extensions):
                os.remove(file_path)

    except Exception as e:
        # os.remove(f'{ds_path}/SPR_neighbors/{n}')
        print(msa_tmpfile.split(SEP)[-1][3:])
        print(e)
        exit()
    finally:
        os.remove(tree_rampath)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SPR neighbors')
    parser.add_argument('--ds_path', '-ds', default=None)
    args = parser.parse_args()
    print("START SPR neighbors generation:", args.ds_path)
    spr_neighbors = generate_neighbors(ds_path=args.ds_path)
    print("DONE SPR neighbors generation:", args.ds_path)