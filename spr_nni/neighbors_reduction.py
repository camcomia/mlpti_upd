from defs_NNI import *
import math, shutil

def reduce_neighbors(spr_folder): 
    spr_neighbor_folders = os.listdir(spr_folder)
    data = []  # List to collect data
    
    # Collect data into a list
    for folder in spr_neighbor_folders:
        path = os.path.join(spr_folder, folder)
        msa_filepath = os.path.join(path, "real_msa.phy")
        stats_filepath = os.path.join(path, "real_msa.phy_phyml_stats_spr.txt")
        res_dict = parse_phyml_stats_output(msa_filepath, stats_filepath)
        data.append([path, res_dict['ll']])
    
    # Convert the collected data into a DataFrame
    df = pd.DataFrame(data, columns=['path', 'likelihood'])
    df['likelihood'] = df['likelihood'].astype(float)
    df.sort_values(by='likelihood', inplace=True, ascending=False)
    
    # Reduce the DataFrame to the top 20%
    rows = df.shape[0]
    num_reduced = math.floor(0.2 * rows)
    df_reduced = df.head(num_reduced)

    # Efficient membership check using a set
    reduced_paths = df_reduced['path']
    reduced_paths_set = set(reduced_paths)
    paths = df['path']
    
    # Count paths to be deleted
    for path in paths:
        if path not in reduced_paths_set:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)  # delete folder
                else:
                    os.remove(path)  # delete file
                print(f"Deleted: {path}")
            else:
                print(f"Path does not exist: {path}")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reduce SPR neighbors')
    parser.add_argument('--spr_folder', '-sf', default=None)
    args = parser.parse_args()
    print(f"START SPR neighbors reduction: {args.spr_folder}")
    reduce_neighbors(args.spr_folder)
    print(f"DONE SPR neighbors reduction: {args.spr_folder}")