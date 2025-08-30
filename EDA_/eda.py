import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

def eda(file_path:str):
    print("Hello we are getting started")
# load the data
    df=pd.read_excel(file_path)
    print("Data loaded")

#printing first 5 and last rows (PPP)
    print(df.head())
    print(df.tail())

#info to know the data
    df.info()

#Renaming columns to lower to avoid key errors
    df.columns=df.columns.str.lower()


eda("C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx")

