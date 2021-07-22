import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class MatDataToCSV():

    def init(self):

        pass

    def convert_mat_tocsv(self):
        working_dir = Path(f"../../imdb_wiki/wiki_crop/")
        mat = scipy.io.loadmat(working_dir / 'wiki.mat')

        instances = mat['wiki'][0][0][0].shape[1]
        columns = ["dob", "photo_taken", "full_path", "gender",\
                "name", "face_location", "face_score", "second_face_score"]
        df = pd.DataFrame(index = range(0,instances), columns = columns)

        for i in mat:
            if i == "wiki":
                current_array = mat[i][0][0]
                for j in range(len(current_array)):
                    df[columns[j]] = pd.DataFrame(current_array[j][0])
        return df

if __name__ == '__main__':
    mat = MatDataToCSV()
    df = mat.convert_mat_tocsv()
    df.to_csv("wiki.csv")