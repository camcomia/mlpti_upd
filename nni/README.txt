pipeline.py
Executes Phyml_BIONJ_startingTrees.py on all multiple sequence alignments (MSAs) located in the training folder to generate starting trees.

collect_features.py
Extracts features relevant to NNI (Nearest Neighbor Interchange) moves for each sequence alignment.
Generates a dataset.csv file per MSA containing the computed features.

machine_learning.py
Combines all individual dataset.csv files into a single DataFrame and performs 10-fold cross-validation for model evaluation.