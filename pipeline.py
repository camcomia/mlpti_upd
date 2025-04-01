import datetime
import subprocess
import argparse
import signal
import sys, os
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
def configure_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

class ProcessManager:
    """Handles parallel execution of feature extraction tasks."""

    def __init__(self):
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)

    def terminate(self, *_):
        """Handles termination signals and kills processes."""
        logging.warning("[!] Interrupt received! Terminating all processes...")
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        except ProcessLookupError:
            pass
        sys.exit(1)


    def execute_command(self, command, folder, task_name):
        """Runs a command and logs output."""
        logging.info(f"STARTED: {task_name} in {folder}")
        log_path = os.path.join(folder, "output.log")

        try:
            with open(log_path, "a", buffering=1) as log:
                process = subprocess.run(
                    command, shell=True, stdout=log, stderr=log,
                    preexec_fn=os.setpgrp if os.name != 'nt' else None,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )
            logging.info(f"DONE: {task_name} in {folder} (Status: {process.returncode})")
        except Exception as e:
            logging.error(f"ERROR during {task_name} in {folder}: {e}")

    def extract_features(self, dataset_path):
        """Runs feature extraction tasks sequentially."""
        logging.info(f"Processing {dataset_path}...")
        start_time = datetime.datetime.now()

        tasks = [
            ("Create Starting Tree", f'python Phyml_BIONJ_startingTrees.py -f {dataset_path}/real_msa.phy'),
            ("Generate SPR Trees", f'python SPR_and_lls.py --dataset_path "{dataset_path}/"'),
            ("Collect Features", f'python collect_features.py --dataset_path "{dataset_path}/"')
        ]
        for task_name, command in tasks:
            self.execute_command(command, dataset_path, task_name)

        logging.info(f"Completed {dataset_path} in {(datetime.datetime.now() - start_time).total_seconds()} seconds")

    def run(self, training_folders, max_workers=6):
        """Runs feature extraction in parallel across dataset folders."""
        dataset_paths = [os.path.join(training_folders, d) for d in os.listdir(training_folders) if os.path.isdir(os.path.join(training_folders, d))]
        logging.info(f"Processing {len(dataset_paths)} datasets in {training_folders}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.extract_features, dataset_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parallel phylogenetic feature extraction.")
    parser.add_argument('--training_folders', '-tf', required=True, help="Path to dataset folders.")
    parser.add_argument('--max_workers', '-mw', type=int, default=6, help="Maximum number of worker threads.")
    parser.add_argument('--log_file', '-lf', type=str, default="pipeline.log", help="Log file name.")
    args = parser.parse_args()

    print("Logfile: {logfile}")
    configure_logging(args.log_file)
    ProcessManager().run(args.training_folders, args.max_workers)
