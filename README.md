# mlpti_upd

## Overview
`mlpti_upd` is a tool for phylogenetic tree inference and machine learning-based optimization. This repository contains scripts for preprocessing phylogenetic data, generating starting trees, and applying learning algorithms.

## Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone <repository-url>
cd mlpti_upd
```

## Setup
Before running the scripts, ensure you have the required dependencies installed and configure the paths to the necessary executables.

### Configuration
1. **Phyml_BIONJ_startingTrees.py**  
   Update the `PHYML_SCRIPT` variable to point to the location of your PhyML executable.

2. **SPR_and_lls.py**  
   Update the `RAXML_NG_SCRIPT` variable to point to the location of your RAxML executable.

## Preprocessing
Run the following scripts in sequence to preprocess the data:
```bash
python Phyml_BIONJ_startingTrees.py -f {real_msa.phy}
python SPR_and_lls.py -ds {training_data/}
python collect_features.py -ds {training_data/}
```
The individual scripts can also be run using pipeline.py over folders using:
```bash
python pipeline.py -tf {training_data/}
```

## Learning Algorithm
Once preprocessing is complete, apply the learning algorithm:
```bash
python RF_learning_algorithm.py -p {project_path/}
```
You can also set PROJECT_PATH in defs_PhyAI.py

## Requirements
- Python 3.x
- PhyML
- RAxML-NG
- Additional Python libraries (install via `requirements.txt` if provided)

## Notes
- Ensure all input files are in the correct format before running the scripts.
- Refer to the comments in each script for additional configuration options.


## Contact
For questions or issues, please contact the repository maintainer.