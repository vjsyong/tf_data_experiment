from datetime import timedelta, datetime
import pandas as pd


def convert_matlab(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    print(python_datetime)

if __name__ == '__main__':
    df = pd.read_csv("wiki.csv")

    paths = df['full_path'] = df['full_path'].str.replace(r'[\[\]\']', '') # Strip [, ], and ' characters

    ages  = []
    for path in paths:
        tokens = path.split("_")
        dob = tokens[1].split("-")[0]
        picture_date = tokens[2].split(".")[0]
        age = int(picture_date) - int(dob)
        ages.append(age)
    
    
    df['age'] = ages

    print(df.head())
    