import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

#load the data
def load_data(file_path:str):
    df=pd.read_excel(file_path)
    df.columns.str.lower()
    print(df.columns)
    return df.drop(columns=['patient id'])
print("data loaded succesfully")

#preprocess
def preprocess_data(df,target_col='level'):
    encoded=LabelEncoder()
    df[target_col]=encoded.fit_transform(df[target_col])
    X=df.drop(colums=[target_col])
    y=df[target_col]
    return X, y,encoded

def main():
    file_path="C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx"
    df=load_data(file_path)
    X,y,encoded=preprocess_data(df)

if __name__=="__main__":
    main()
