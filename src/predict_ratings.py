"""
Cognifyz Technologies - Machine Learning Internship
Task 1: Predict Restaurant Ratings

Aim:
    Build a machine learning model to predict the aggregate rating of a
    restaurant based on other features in the dataset.

This script:
    1. Loads and preprocesses the dataset (missing values, encoding).
    2. Splits data into train/test sets.
    3. Trains two regression models (Linear Regression, Decision Tree).
    4. Evaluates them using MSE and R-squared.
    5. Reports the most influential features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def preprocess(df):
    df = df.copy()

    # Drop rows with no target value (aggregate rating)
    df = df.dropna(subset=["Aggregate rating"])

    # Drop columns that are identifiers / free text / redundant for modeling
    drop_cols = [
        "Restaurant ID", "Restaurant Name", "Address", "Locality Verbose",
        "Rating color", "Rating text", "Switch to order menu"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Handle missing values
    # Numeric columns -> fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns -> fill with mode / 'Unknown'
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    # Encode categorical variables with LabelEncoder
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def train_and_evaluate(df):
    X = df.drop(columns=["Aggregate rating"])
    y = df["Aggregate rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # ---- Linear Regression ----
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    results["Linear Regression"] = {
        "model": lin_model,
        "mse": mean_squared_error(y_test, lin_pred),
        "r2": r2_score(y_test, lin_pred),
    }

    # ---- Decision Tree Regression ----
    tree_model = DecisionTreeRegressor(random_state=42, max_depth=10)
    tree_model.fit(X_train, y_train)
    tree_pred = tree_model.predict(X_test)
    results["Decision Tree Regression"] = {
        "model": tree_model,
        "mse": mean_squared_error(y_test, tree_pred),
        "r2": r2_score(y_test, tree_pred),
    }

    return results, X, X_test, y_test


def report_feature_importance(results, X):
    tree_model = results["Decision Tree Regression"]["model"]
    importances = pd.Series(tree_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nTop influential features (Decision Tree):")
    print(importances.head(10))

    plt.figure(figsize=(8, 6))
    importances.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances - Restaurant Rating Prediction")
    plt.xlabel("Importance")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved feature importance chart to: {fig_path}")

    return importances


def main():
    df = load_data()
    df_processed, encoders = preprocess(df)

    results, X, X_test, y_test = train_and_evaluate(df_processed)

    print("\nModel Performance:")
    report_lines = ["Task 1: Predict Restaurant Ratings - Model Report", "=" * 50, ""]
    for name, res in results.items():
        line = f"{name}: MSE = {res['mse']:.4f}, R-squared = {res['r2']:.4f}"
        print(line)
        report_lines.append(line)

    importances = report_feature_importance(results, X)
    report_lines.append("\nTop 10 influential features (Decision Tree):")
    for feat, imp in importances.head(10).items():
        report_lines.append(f"  {feat}: {imp:.4f}")

    report_path = os.path.join(OUTPUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved model report to: {report_path}")


if __name__ == "__main__":
    main()
