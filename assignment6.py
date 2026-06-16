# ==========================================
# Automated Customer Loan Eligibility Approval Predictor
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ==========================================
# STEP 1: Load Dataset
# ==========================================

try:
    df = pd.read_csv("loan_data.csv")
    print("Dataset Loaded Successfully!\n")
except FileNotFoundError:
    print("loan_data.csv not found!")
    exit()

# ==========================================
# STEP 2: Display Dataset Information
# ==========================================

print("First 5 Records:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# STEP 3: Handle Missing Values
# ==========================================

# Fill numeric missing values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical missing values
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==========================================
# STEP 4: Encode Categorical Features
# ==========================================

encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

print("\nEncoded Dataset:")
print(df.head())

# ==========================================
# STEP 5: Feature Selection
# ==========================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

print("\nFeatures:")
print(X.columns.tolist())

# ==========================================
# STEP 6: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# STEP 7: Train Decision Tree Model
# ==========================================

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# ==========================================
# STEP 8: Predictions
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# STEP 9: Evaluation
# ==========================================

cm = confusion_matrix(y_test, predictions)

accuracy = accuracy_score(y_test, predictions)

report = classification_report(y_test, predictions)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

print("\n==============================")
print("ACCURACY")
print("==============================")
print(f"{accuracy*100:.2f}%")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(report)

# ==========================================
# STEP 10: Feature Importance
# ==========================================

importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(feature_importance_df)

# ==========================================
# STEP 11: Feature Importance Visualization
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================
# STEP 12: Predict New Customer
# ==========================================

print("\n==============================")
print("NEW CUSTOMER PREDICTION")
print("==============================")

sample_customer = {
    "Gender": "Male",
    "Age": 35,
    "Marital_Status": "Married",
    "Education": "Graduate",
    "Employment_Status": "Employed",
    "Annual_Income": 80000,
    "Loan_Amount": 250000,
    "Credit_Score": 780,
    "Existing_Loan": "No",
    "Property_Area": "Urban"
}

sample_df = pd.DataFrame([sample_customer])

# Encode categorical values
for col in sample_df.columns:
    if col in encoders:
        sample_df[col] = encoders[col].transform(sample_df[col])

prediction = model.predict(sample_df)

result = encoders["Loan_Status"].inverse_transform(prediction)

print("\nCustomer Information:")
print(sample_customer)

print("\nPrediction:")
print(result[0])

# ==========================================
# STEP 13: Save Feature Importance Report
# ==========================================

feature_importance_df.to_csv(
    "feature_importance_report.csv",
    index=False
)

print("\nFeature importance report saved as:")
print("feature_importance_report.csv")

print("\nProject Completed Successfully!")