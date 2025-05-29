# general imports
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

SEP = "/"

######## CHANGEME ########
PROJECT_PATH = "/scratch/ctcomia/mlpti/training_data/groups/itay_mayrose/danaazouri/PhyAI/submission_data/training_data_all"
##########################


DATA_PATH = SEP.join([PROJECT_PATH, "training_data", ""])
CODE_PATH = SEP.join([PROJECT_PATH, "code", ""])



###############################################################################
######################## global vars for common strings #######################
###############################################################################
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
SUMMARY_PER_DS = "{}ds_summary_{}_{}_step{}.csv"
TREES_PER_DS = "{}newicks_step{}.csv"
LEARNING_DATA = "learning_{}_step{}.csv"
DATA_WITH_PREDS = "with_preds_merged_{}.csv"
SCORES_PER_DS = "scores_per_ds_{}.csv"
###############################################################################



###############################################################################
############################ features & label vars ############################
###############################################################################
LABEL = "d_ll_{}"
FEATURES = OrderedDict([("bl", "edge_length"), ("longest", "longest_branch") ,
						("tbl_p","tbl_pruned"),("tbl_r","tbl_remaining"),
						("longest_p","longest_pruned"),("longest_r","longest_remaining"),
						("top_dist","topology_dist_between"), ("bl_dist","tbl_dist_between"),   # only for rgft
						("res_bl", "res_tree_edge_length"),                                     # only for rgft
						("group_id", "orig_ds_id"), ("group_tbl","orig_ds_tbl"),
						("ntaxa_p", "num_taxa_prune"), ("ntaxa_r", "num_taxa_rgft")])


FEATURES_RGFT_ONLY = ["top_dist", "bl_dist", "res_bl"]
FEATURES_RGFT = [feature for key, feature in FEATURES.items()]
FEATURES_PRUNE = [feature for key, feature in FEATURES.items()]
[FEATURES_PRUNE.remove(FEATURES[f]) for f in FEATURES_RGFT_ONLY]
FEATURES_SHARED = ["orig_ds_id", "orig_ds_tbl", "longest_branch"]
merged_prune, merged_rgft = FEATURES_PRUNE.copy(), FEATURES_RGFT.copy()
[merged_prune.remove(f) for f in FEATURES_SHARED], [merged_rgft.remove(f) for f in FEATURES_SHARED]
FEATURES_MERGED = FEATURES_SHARED + [feature + "_prune" for feature in merged_prune] + [feature + "_rgft" for feature in merged_rgft]
###############################################################################



###############################################################################
##### manually define the type, to enhance performance of pandas read_csv #####
###############################################################################
list_str = ['Source', 'path', 'prune_name', 'rgft_name']
list_int = ['orig_ntaxa', 'ntaxa', 'nchars', 'orig_ds_id']
list_float = ['orig_ds_ll', 'll', 'd_ll_prune', 'edge_length_prune', 'longest_branch', 'ntaxa_prunned_prune', 'tbl_pruned_prune', 'longest_pruned_prune', 'ntaxa_remaining_prune', 'tbl_remaining_prune', 'longest_remaining_prune', 'orig_ds_tbl', 'd_ll_rgft', 'edge_length_rgft', 'ntaxa_prunned_rgft', 'tbl_pruned_rgft', 'longest_pruned_rgft', 'ntaxa_remaining_rgft', 'tbl_remaining_rgft', 'longest_remaining_rgft', 'topology_dist_between_rgft', 'tbl_dist_between_rgft', 'd_ll_merged', 'edge_length', 'ntaxa_prunned', 'tbl_pruned', 'longest_pruned', 'ntaxa_remaining', 'tbl_remaining', 'longest_remaining', 'topology_dist_between', 'tbl_dist_between', "res_tree_bl_rgft"]

types_dict = {}
for e in list_str:
	types_dict[e] = np.object_
for e in list_int:
	types_dict[e] = np.int32
for e in list_float:
	types_dict[e] = np.float32
###############################################################################
