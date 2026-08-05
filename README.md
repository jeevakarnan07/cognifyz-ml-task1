# Task 1 — Predict Restaurant Ratings
Cognifyz Technologies · Machine Learning Internship

## Aim
Build a machine learning model to predict the aggregate rating of a
restaurant based on other features in the dataset.

## Dataset
`data/Dataset.csv` — 9,551 restaurants with features such as location,
cuisines, price range, votes, and the target column `Aggregate rating`.

## Procedure
1. **Preprocessing** (`src/predict_ratings.py`)
   - Dropped identifier / free-text columns not useful for modeling
     (Restaurant ID, Name, Address, etc.).
   - Filled missing numeric values with the median, missing categorical
     values with `"Unknown"`.
   - Encoded categorical variables using `LabelEncoder`.
   - Split data 80/20 into training and testing sets.
2. **Modeling**
   - Trained a **Linear Regression** model as a baseline.
   - Trained a **Decision Tree Regressor** (max_depth=10).
3. **Evaluation**
   - Compared models using Mean Squared Error (MSE) and R-squared (R²)
     on the held-out test set.
4. **Interpretation**
   - Extracted feature importances from the Decision Tree model to see
     which features most influence the predicted rating.

## Result
| Model | MSE | R² |
|---|---|---|
| Linear Regression | ~1.56 | ~0.31 |
| Decision Tree Regression | ~0.11 | ~0.95 |

The **Votes** column is by far the most influential feature, followed by
location (Longitude/Latitude) and Cuisines. Full numbers are written to
`outputs/model_report.txt` and a chart is saved to
`outputs/feature_importance.png` each time the script runs.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/predict_ratings.py
```

Outputs (`model_report.txt`, `feature_importance.png`) are written to the
`outputs/` folder.

## Project structure
```
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
