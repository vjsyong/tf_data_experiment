import pandas as pd
import pathlib
import os

skip_list = []
sub_dir = "test"

process_dir = f"appa-real/{sub_dir}"
dir = pathlib.Path(f"../{process_dir}")

df = pd.read_csv(dir / f"../gt_avg_{sub_dir}.csv")

file_list = df['file_name'].tolist()

for file in file_list:
    if not os.path.isfile(pathlib.Path(str(dir) + "_aligned") / file):
        skip_list.append(file)

df = df[~df['file_name'].isin(skip_list)]
df.to_csv(f"../appa-real/gt_avg_{sub_dir}_aligned.csv")