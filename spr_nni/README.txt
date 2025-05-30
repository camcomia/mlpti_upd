##############################################################################################################
spr_nni.batch
Runs the full pipeline for tree generation, feature extraction, and dataset preparation per alignment:

Starting Tree Generation
Uses Phyml_BIONJ_startingTrees.py to generate starting trees for each MSA.

SPR Neighbor Generation
Applies generate_SPR_trees.py to create SPR neighbors from the starting trees.

Neighbor Reduction
Filters the top 20% of SPR neighbors with the highest likelihood scores using neighbors_reduction.py.

NNI Neighbor Generation
Generates all possible NNI neighbors from the top SPR trees via generate_NNI_trees.py.

Dataset Merging
Merges feature datasets into a single file per sequence alignment.

##############################################################################################################
training.batch
Performs machine learning with 10-fold cross-validation using the merged datasets for training and evaluation.

##############################################################################################################
validation.batch
Executes the trained machine learning model on a separate validation set for performance assessment.