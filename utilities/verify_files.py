import pandas as pd
import pathlib
import os

skip_list = []

process_dir = f"imdb_wiki/imdb_crop/"
dir = pathlib.Path(f"../{process_dir}")

df = pd.read_csv(dir / "imdb.csv")
print(df.shape[0])

file_list = df['full_path'].str.replace(r'[\[\]\']', '').tolist()

for file in file_list:
    # print(dir / file)
    if not os.path.isfile(dir / file):
        # print(file)
        skip_list.append(file)

print(len(skip_list))
df = df[~df['full_path'].str.replace(r'[\[\]\']', '').isin(skip_list)]
df.to_csv(dir / f"imdb_processed.csv")