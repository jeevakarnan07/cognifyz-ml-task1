# Restaurant Rating Prediction

Machine learning regression project that predicts restaurant aggregate ratings from restaurant attributes using scikit-learn.

## Project Summary

This project builds and compares regression models for predicting restaurant ratings from a dataset of 9,551 restaurants. It demonstrates a complete ML workflow: data cleaning, categorical encoding, train/test evaluation, model comparison, and feature-importance analysis.

## Models

- Linear Regression — baseline model
- Decision Tree Regressor — non-linear model

## Evaluation

Models are evaluated on a held-out test set using:

- Mean Squared Error (MSE)
- R² score

The current Decision Tree baseline substantially outperforms Linear Regression on the supplied dataset. See `outputs/model_report.txt` for the run-specific metrics.

## Key Insight

Votes are the strongest predictive signal in the current feature set, with geographic and cuisine-related variables also contributing. The project should treat the high tree-model R² as a dataset-specific result rather than a guarantee of real-world performance.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib

## Project Structure

```text
cognifyz-ml-task1/
├── data/
│   └── Dataset.csv
├── src/
│   └── predict_ratings.py
├── outputs/
│   ├── model_report.txt
│   └── feature_importance.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python src/predict_ratings.py
```

## Recommended Next Improvements

1. Replace ordinal-style `LabelEncoder` treatment of nominal input features with a `ColumnTransformer` and `OneHotEncoder` where appropriate.
2. Add MAE and RMSE alongside MSE/R².
3. Add cross-validation and compare Random Forest / Gradient Boosting regressors.
4. Add a small prediction interface or Streamlit demo.
5. Save the trained model and preprocessing pipeline for reproducible inference.

## Resume Description

**Restaurant Rating Prediction | Python, scikit-learn, pandas** — Built regression models to predict restaurant ratings from 9,551 records; implemented preprocessing, categorical encoding, model comparison, evaluation with MSE/R², and feature-importance analysis.

## Author

**Jeeva Karnan** · B.Tech Artificial Intelligence & Data Science
