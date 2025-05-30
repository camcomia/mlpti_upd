import pandas as pd
import glob, argparse, shutil, os, datetime

def clear_files(ds_path):
    msa_folders = os.listdir(ds_path)
    for msa_folder in msa_folders:
        ds_csv = glob.glob(f"{ds_path}SPR_neighbors/*/dataset.csv", recursive=True)
        # print(ds_csv)
        df = pd.DataFrame()
        for ds in ds_csv:
            df_ds = pd.read_csv(ds)
            #print(df_ds.shape)
            df = pd.concat([df, df_ds])
        df.to_csv(f"{ds_path}dataset.csv", index=False)
    # print (f"SPR Folder path: {ds_path}SPR_neighbors")
    if (os.path.exists(f"{ds_path}SPR_neighbors")):
        print (f"Deleting {ds_path}SPR_neighbors")
        shutil.rmtree(f"{ds_path}SPR_neighbors")
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete files')
    parser.add_argument('--ds_path', '-ds', default=None)
    args = parser.parse_args()
    print("START merging datasets:", args.ds_path + "SPR_neighbors at", datetime.datetime.now())
    spr_neighbors = clear_files(ds_path=args.ds_path)
    print("DONE merging datasets:", args.ds_path + "SPR_neighbors at", datetime.datetime.now())
