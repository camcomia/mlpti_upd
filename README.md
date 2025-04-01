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
python Phyml_BIONJ_startingTrees.py
python SPR_and_lls.py
python collect_features.py
```

## Learning Algorithm
Once preprocessing is complete, apply the learning algorithm:
```bash
python RF_learning_algorithm.py
```

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