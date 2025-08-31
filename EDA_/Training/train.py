import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

#load the data
def load_data(file_path:str):

    df=pd.read_excel(file_path)
    df.columns=df.columns.str.lower()
    print(df.columns)
    return df.drop(columns=['patient id'])
print("data loaded succesfully")

#preprocess
def preprocess_data(df, target_col='level'):
    encoder = LabelEncoder()
    
    # Encode the target column 
    df[target_col] = encoder.fit_transform(df[[target_col]])
    
    # Split into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, encoder

def Logistic_Reg_train(X_train,y_test,y_train,X_test,encoder):
    model=LogisticRegression(max_iter=200,multi_class="multinomial")
    model.fit(X_train,y_train)
    y_predicted=model.predict(X_test)
    print("Done logistic Regression")
    print(classification_report(y_test,y_predicted,target_names=encoder.classes_))
    return model

def Random_forest_train(X_train,y_test,y_train,X_test):
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_rep)


def main():
    file_path="C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx"
    df=load_data(file_path)
    X,y,encoder=preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,train_size=0.3, test_size=0.7, random_state=42
)
    Logistic_Reg_train(X_train,y_test,y_train,X_test,encoder)
    Random_forest_train(X_train,y_test,y_train,X_test)


if __name__=="__main__":
    main()
