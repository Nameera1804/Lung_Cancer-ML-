import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Import os Library

def load_data(file_path:str):
    print("Hello we are getting started")
# load the data
    data=pd.read_excel(file_path)
    df=pd.DataFrame(data)
    print(type(df))
    print("Loaded successfully")

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
    return df

#Analysis using histogram 
#univariate to identify level distribution
def uni_analysis(df:pd.DataFrame,my_path:str):
    plt.figure(figsize=(9,7))
    sns.countplot(x='level',data=df,order=['Low','Medium','High'] ,palette='viridis')
    plt.title('distribution of level')
    plt.ylabel=('No. of Patients')
    plt.savefig(os.path.join(my_path, 'level_distribution.png'))
    plt.close()

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
    plt.tight_layout()
    plt.savefig(os.path.join(my_path, 'feature distribution.png'))
    print('Saved feature distribution')
    plt.close()

#How features are distributed with Target (level)
    # features1=df.drop('patient id',axis=1).columns
def bivariate_analysis(df:pd.DataFrame,my_path:str):
    plt.figure(figsize=(30,40))
    for i,column in enumerate(df):
        plt.subplot(8,3,i+1)
        sns.boxplot(x='level',data=df, y=column,order=['Low','Medium','High'])
        plt.title(f'Distribution of{column}')
        # plt.xlabel('')
        # plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(my_path, 'feature vs target .png'))
    plt.close()
    print('Saved feature vs target distribution')

#Correlation using spearman correlation
def multivariate_analysis(df:pd.DataFrame,my_path:str):
    print("Starting correlation")
    df_corr=df.copy()
    if 'patient id' in df_corr.columns:
        df_corr = df_corr.drop('patient id', axis=1)

    # ord_enc = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
    # df_corr[['level']] = ord_enc.fit_transform(df_corr[['level']])
    plt.figure(figsize=(30,40))
    df_corr['level']=df_corr['level'].map({'Low':0,'Medium':1,'High':2})
    correlation_matrix = df_corr.corr(method='spearman')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig(my_path +'correlation.png')
    plt.show()
    print("saved")
    # else:
    #     print('pateint id is not dropped')

def main():
    file_path="C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx"
    cleaned_df=load_data(file_path)
    main_folder = r"C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/EDA_"   # main path
# output folder path inside main folder
    my_path = os.path.join(main_folder, "output")
    # Create the output folder if it doesn't exist
    os.makedirs(my_path, exist_ok=True)
    uni_analysis(cleaned_df,my_path)
    bivariate_analysis(cleaned_df,my_path)
    multivariate_analysis(cleaned_df,my_path)
    print("EDA Complete")

if __name__=='__main__':
    main()

    