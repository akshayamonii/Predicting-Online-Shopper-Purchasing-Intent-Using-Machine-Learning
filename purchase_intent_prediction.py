# FINAL MACHINE LEARNING PROJECT: ONLINE SHOPPER PURCHASING INTENTION

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# LOAD DATA
df = pd.read_csv("C:/Users/aksha/Downloads/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


# EXPLORATORY DATA ANALYSIS (EDA)

# 1. Purchase vs Non-Purchase
plt.figure(figsize=(6,4))
sns.countplot(x=df['Revenue'])
plt.title("Purchase vs Non-Purchase Distribution")
plt.show()

# 2. Correlation Heatmap (NUMERIC COLUMNS ONLY)
plt.figure(figsize=(14,10))
numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# 3. Purchase Rate by Month
plt.figure(figsize=(10,6))
df.groupby("Month")["Revenue"].mean().plot(kind='bar')
plt.title("Purchase Rate by Month")
plt.ylabel("Average Purchase Rate")
plt.show()

# 4. Purchase Rate by Visitor Type
plt.figure(figsize=(8,5))
sns.barplot(x="VisitorType", y="Revenue", data=df)
plt.title("Purchase Rate by Visitor Type")
plt.show()

# 5. Product Duration vs Purchase
plt.figure(figsize=(10,6))
sns.boxplot(x="Revenue", y="ProductRelated_Duration", data=df)
plt.title("Product Page Duration vs Purchase")
plt.show()


# PREPROCESSING

# Convert boolean columns to integers
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=["Month", "VisitorType"], drop_first=True)

# Separate features and target
X = df_encoded.drop("Revenue", axis=1)
y = df_encoded["Revenue"]

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MODEL 1: LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

y_pred_lr = log_model.predict(X_test)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))


# MODEL 2: RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# FEATURE IMPORTANCE PLOT

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-12:]  

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

print("\nProject completed successfully.")
