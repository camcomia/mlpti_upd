#!/bin/bash
# This script is used to run phyml on multiple alignments for starting tree generation from msa files
# Before running this script, check if all the binary files are properly pointed to the right directories
# PHYML_SCRIPT in Phyml_BIONJ_startingTrees.py
# RAXML_NG_SCRIPT in run_spr_on_tree.py


# Directory containing the folders with original real_msa.phy
DATASET_DIR="/home/ctcomia/mlpti/training_data/hybrid_testing/20_data"

# Find all real_msa.phy
MSA_FILES=($(find "${DATASET_DIR}" -name 'real_msa.phy'))

# Check if files were found
if [ ${#MSA_FILES[@]} -eq 0 ]; then
    echo "No real_msa.phy files found in ${DATASET_DIR}"
    exit 1
fi

echo "Found ${#MSA_FILES[@]} msa files to process."

# Loop through each NNI tree
for MSA_FILE in "${MSA_FILES[@]}"; do
    echo "Processing alignment: $MSA_FILE"
    # Run the Python script
    python Phyml_BIONJ_startingTrees.py \
        -f "${MSA_FILE}" \

    # Optional: Add error checking based on the python script's exit code
    if [ $? -ne 0 ]; then
        echo "Error processing $MSA_FILE. Check logs."
        # Uncomment line below if you want script to halt upon error, if not comment line out
        # exit 1 # Exit on first error
    fi
    echo "" # Add a newline for readability
done

echo "Finished processing all real_msa.phy files."
