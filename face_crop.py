from deepface import DeepFace
import pathlib
import cv2
from joblib import Parallel, delayed
import pandas as pd 
from tqdm import tqdm

# Loop through ChaLearn 2016 dataset to crop and align faces for model training

process_dir = "appa-real/test"

dir = pathlib.Path(f"../{process_dir}")

skip_list = []

def align_image(path):
    if "_face" in str(path):
        return

    filename = pathlib.Path(f"../{process_dir}_aligned") / str(path).split("/")[-1]
    try:
        aligned_face = DeepFace.detectFace(str(path), detector_backend = 'dlib')
    except:
        skip_list.append(str(path).split("/")[-1]) # Append to skip list to remove from entry later
        return

    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(filename), aligned_face * 255)

Parallel(n_jobs=20)(delayed(align_image)(path) for path in tqdm(dir.glob("*")))


print("skip", skip_list)

df = pd.DataFrame(skip_list, columns=["colummn"])
df.to_csv('list.csv', index=False)
