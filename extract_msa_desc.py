import os
import sys
import glob

def extract_lines(base_folder, output_csv="output.csv"):
    folders = [f for f in glob.glob(os.path.join(base_folder, "*/")) if os.path.isdir(f)]
    
    print("TAXA\tSITES\tFOLDER\n-------------------------")
    
    for folder in folders:
        folder_name = folder.rstrip('/').split('/')[-1] 
        file_path = os.path.join(folder, "real_msa.phy")
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                taxa, site = file.readline().strip().split(" ")
#                print(taxa + "\t" + site + "\t" + folder_name) 
                print(f"{taxa}\t{site}\t/{folder_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_msa_desc.py <base_folder>")
        sys.exit(1)

    base_folder = sys.argv[1]
    extract_lines(base_folder)
