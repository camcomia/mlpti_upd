#!/bin/bash

# Directory containing the original MSA, stats file, and the NNI subdirectory
DATASET_DIR="/home/ctcomia/mlpti/training_data/hybrid_testing/20_data"

# Check if the directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory does not exist: $DATASET_DIR"
    exit 1
fi
echo "Dataset directory: $DATASET_DIR"

# Find the individual MSA and stats file
MSA_FOLDER=($(find ${DATASET_DIR}/* -type d ! -name NNI))
if [ ${#MSA_FOLDER[@]} -eq 0 ]; then
    echo "No MSA folder found in ${DATASET_DIR}"
    exit 1
fi

echo "Found ${#MSA_FOLDER[@]} MSA folders."

for msa in "${MSA_FOLDER[@]}"; do
    echo "Found MSA directory: $msa"

    MSA_FILE="${msa}/real_msa.phy"
    STATS_FILE="${msa}/real_msa.phy_phyml_stats_bionj.txt" # Adjust if filename differs

    # Find all NNI output trees
    NNI_TREES=($(find "${msa}/NNI" -name 'optimized_*.raxml.bestTree'))

    # Check if files were found
    if [ ${#NNI_TREES[@]} -eq 0 ]; then
        echo "No NNI output trees found in ${msa}/NNI"
        exit 1
    fi

    echo "Found ${#NNI_TREES[@]} NNI trees to process."

    # Loop through each NNI tree
    for nni_tree_file in "${NNI_TREES[@]}"; do
        # Extract the index number (e.g., 1 from optimized_1.raxml.bestTree)
        index=$(basename "$nni_tree_file" | sed -n 's/optimized_\([0-9]*\)\.raxml\.bestTree/\1/p')

        if [ -z "$index" ]; then
            echo "Warning: Could not extract index from $nni_tree_file. Skipping."
            continue
        fi

        # Define a unique output directory for this NNI tree's SPR results
        OUTPUT_SPR_DIR="${msa}/Hybrid_NNI_SPR/${index}"

        echo "-----------------------------------------------------"
        echo "Processing NNI Tree: $nni_tree_file"
        echo "Outputting SPR results to: $OUTPUT_SPR_DIR"
        echo "-----------------------------------------------------"

        # Run the Python script
        python run_spr_on_tree.py \
            --input_tree "$nni_tree_file" \
            --msa "$MSA_FILE" \
            --params_stats_file "$STATS_FILE" \
            --output_dir "$OUTPUT_SPR_DIR"

        # Optional: Add error checking based on the python script's exit code
        if [ $? -ne 0 ]; then
            echo "Error processing $nni_tree_file. Check logs."
            # Decide whether to continue or exit the loop
            # exit 1 # Exit on first error
        fi
        echo "" # Add a newline for readability
    done
done

echo "Finished processing all NNI trees."