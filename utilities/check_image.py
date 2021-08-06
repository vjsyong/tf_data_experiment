import imghdr
import os, pathlib
import glob

process_dir = f"training_data/imdb_wiki/imdb_crop/"
dir = pathlib.Path(f"../{process_dir}")
# l_FileNames = os.listdir("../training_data/imdb_wiki/wiki_crop/")

# for image in l_FileNames:
#     if not imghdr.what(image) == "jpeg":
#         # l_FileNames.remove(image)
#         print(image)

for image in dir.rglob('*.jpg'):
    size = os.path.getsize(image)
    if not size > 0:
        # l_FileNames.remove(image)
        print(image)
