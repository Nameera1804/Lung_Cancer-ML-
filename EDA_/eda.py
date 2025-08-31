import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

def eda(file_path:str):
    print("Hello we are getting started")
# load the data
    data=pd.read_excel(file_path)
    df=pd.DataFrame(data)
    print(type(df))
    if isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        print("Data loaded")

#printing first 5 and last rows (PPP)
    print(df.head())
    print(df.tail())

#info to know the data
    df.info()

#Renaming columns to lower to avoid key errors
    df.columns=df.columns.str.lower()
    print(df.columns)

#drop the patient id bcz its not  a feature
    if 'patient id' in df.columns:
        df=df.drop( 'patient id',axis=1)
        print('patient id dropped')

#Analysis using histogram 
#univariate to identify level distribution
    plt.figure(figsize=(9,7))
    sns.countplot(x='level',data=df,order=['Low','Medium','High'] ,palette='viridis')
    plt.title('distribution of level')
    plt.ylabel=('No. of Patients')
    plt.savefig('level_distribution.png')

# # we can also do this 
#     print(df.level.value_counts())

#Analyzing all the features
    
    features=df.drop('level',axis=1)
    print(type(features))
    plt.figure(figsize=(30,40))
    for i,column in enumerate(features):
        plt.subplot(8,3,i+1)
        sns.histplot(df[column],kde=True,bins=df[column].nunique())
        plt.title(f'Distribution of{column}')
        # plt.xlabel('')
        # plt.ylabel('')
    plt.tight_layout()
    plt.savefig('feature distribution.png')
    print('Saved feature distribution')

#How features are distributed with Target (level)
    # features1=df.drop('patient id',axis=1).columns
    plt.figure(figsize=(30,40))
    for i,column in enumerate(df):
        plt.subplot(8,3,i+1)
        sns.boxplot(x='level',data=df, y=column,order=['Low','Medium','High'])
        plt.title(f'Distribution of{column}')
        # plt.xlabel('')
        # plt.ylabel('')
    plt.tight_layout()
    plt.savefig('feature vs target .png')
    print('Saved feature vs target distribution')

#Correlation using spearman correlation
    print("Starting correlation")
    df_corr=df.copy()
    if 'patient id' in df_corr.columns:
        df_corr = df_corr.drop('patient id', axis=1)

    ord_enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
    df_corr[['level']] = ord_enc.fit_transform(df_corr[['level']])

    correlation_matrix = df_corr.corr(method='spearman')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation.png')
    # else:
    #     print('pateint id is not dropped')






eda("C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx")

