import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class MatDataToCSV():

    def init(self):

        pass

    def convert_mat_tocsv(self, name):
        working_dir = Path(f"../imdb_wiki/{name}_crop/")
        print(working_dir / f'{name}.mat')
        mat = scipy.io.loadmat(working_dir / f'{name}.mat')

        instances = mat[name][0][0][0].shape[1]
        columns = ["dob", "photo_taken", "full_path", "gender",\
                "name", "face_location", "face_score", "second_face_score"]
        df = pd.DataFrame(index = range(0,instances), columns = columns)

        for i in mat:
            if i == name:
                current_array = mat[i][0][0]
                for j in range(len(current_array)):
                    try:
                        df[columns[j]] = pd.DataFrame(current_array[j][0])
                    except:
                        print("column failed")
        df.to_csv(working_dir / f"{name}.csv")
        return df

if __name__ == '__main__':
    mat = MatDataToCSV()
    df = mat.convert_mat_tocsv("wiki")
    df = mat.convert_mat_tocsv("imdb")
