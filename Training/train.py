import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

def Logistic_Reg_train(X_train,X_test,y_train,y_test,encoder):
    model=LogisticRegression(max_iter=100,multi_class="multinomial")
    model.fit(X_train,y_train)
    y_predicted=model.predict(X_test)
    print("Done logistic Regression")
    print(classification_report(y_test,y_predicted,target_names=encoder.classes_))
    return model

print("working with other model")


def Random_forest_train(X_train, X_test, y_train, y_test):
    # Initialize model
    print("Random forest training started")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # Train
    rf_classifier.fit(X_train, y_train)
    # Predict
    y_pred = rf_classifier.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    # Feature importances
    feature_importances = rf_classifier.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot and save BEFORE plt.show()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png")  # saves properly
    plt.show()
    plt.close()

    # Save as CSV for later analysis
    feature_importance_df.to_csv("feature_importance.csv", index=False)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nRandom Forest Classification Report:\n", classification_rep)
    print("\nFeature importances saved to 'feature_importance.csv' and 'feature_importance.png'.")
    return rf_classifier
    

def main():
    file_path="C:/Users/nameera.zuha/OneDrive - SLK SOFTWARE PRIVATE LIMITED/Documents/Lung_Cancer_classif/raw_data/cancer patient data sets.xlsx"
    df=load_data(file_path)
    X,y,encoder=preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=42
)
    Logistic_Reg_train(X_train,X_test,y_train,y_test,encoder)
    print("working with other model")
    Random_forest_train(X_train,X_test,y_train,y_test)


if __name__=="__main__":
    main()
