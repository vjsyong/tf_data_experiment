import pandas as pd
import pathlib
import os

skip_list = []
sub_dir = "test"

process_dir = f"ChaLearn_aligned/"
dir = pathlib.Path(f"../{process_dir}")

df = pd.read_csv(dir / "train_gt.csv")

file_list = df['image'].tolist()

for file in file_list:
    if not os.path.isfile(dir / file):
        skip_list.append(file)

df = df[~df['image'].isin(skip_list)]
df['mean'] = df['mean'].round(0)
df = df.drop('stdv', 1)
df.to_csv(f"train_gt_aligned.csv")