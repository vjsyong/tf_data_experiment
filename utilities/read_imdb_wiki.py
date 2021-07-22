import scipy.io
from pathlib import Path
import pandas as pd

working_dir = Path(f"../../imdb_wiki/wiki_crop/")
mat = scipy.io.loadmat(working_dir / 'wiki.mat')

mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()}) # compatible for both python 2.x and python 3.x

data.to_csv("wiki.csv")