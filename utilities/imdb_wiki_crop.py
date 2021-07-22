
from pathlib import Path
from deepface import DeepFace
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm



# for file_path in Path(working_dir).glob('**/*.jpg'):
#     print(file_path) # do whatever you need with these files

def align_image(path):
    save_dir = '/'.join(str(path).split("/")[0:5])
    save_dir = save_dir.replace("imdb_crop", "imdb_crop_aligned")
    filename = Path(str(path).replace('imdb_crop', 'imdb_crop_aligned'))
    
    save_dir = Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        aligned_face = DeepFace.detectFace(str(path), detector_backend = 'dlib')
    except:
        # skip_list.append(str(path).split("/")[-1]) # Append to skip list to remove from entry later
        return

    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(filename), aligned_face * 255)

# if __name__ == '__main__':
working_dir = Path(f"../../imdb_wiki/imdb_crop/")
Parallel(n_jobs=20)(delayed(align_image)(path) for path in tqdm(working_dir.glob('**/*.jpg')))