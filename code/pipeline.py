#!/usr/bin/env python

import datetime, subprocess, argparse, os
from concurrent.futures import ProcessPoolExecutor

def execute_command(command, log_file, folder):
    # Generic method to execute shell command and logs output/errors into specified file (log_file).
    print(datetime.datetime.now(), f"STARTED: {command} in {folder}")
    log_path = os.path.join(folder, log_file)
    try:
        with open(log_path, "a") as log:
            process = subprocess.Popen(command, shell=True, stdout=log, stderr=log)
            process.wait()
            status = process.returncode
        print(datetime.datetime.now(), f"DONE: {command} in {folder} (Status: {status})")
    except Exception as e:
        print(datetime.datetime.now(), f"ERROR in {command} in {folder}: {e}")
        status = -1
    return status

def executePhyml(folder):
    return execute_command(f'python Phyml_BIONJ_startingTrees.py -f {folder}/real_msa.phy', "generate_phyml_logs.txt", folder)

def executeSPR(folder):
    return execute_command(f'python ./codeNNI/SPR_and_lls.py --dataset_path "{folder}/"', "generate_spr_logs.txt", folder)

def executeCollection(folder):
    return execute_command(f'python ./codeNNI/collect_features.py --dataset_path "{folder}/"', "generate_collection_logs.txt", folder)

def extractFeatures(dataset_path):
    # Executes all processes for a given dataset sequentially.
    print("START extractFeatures:", dataset_path, "at", datetime.datetime.now())
    if os.path.exists(dataset_path):
        executePhyml(dataset_path)
        # executeSPR(dataset_path)
        # executeCollection(dataset_path)
    print("DONE extractFeatures:", dataset_path, "at", datetime.datetime.now())

def main(training_folders):
    print("Beginning processing of datasets located at:", training_folders, "at", datetime.datetime.now)

    # Collect valid dataset paths
    dataset_paths = [os.path.join(training_folders, f) for f in os.listdir(training_folders) if os.path.isdir(os.path.join(training_folders, f))]

    # Use multiprocessing to run extractFeatures in parallel
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count()-2)) as executor: #adjust max workers accordingly
        executor.map(extractFeatures, dataset_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform all SPR moves')
    parser.add_argument('--training_folders', '-tf', required=True)
    args = parser.parse_args()

    main(args.training_folders)

    