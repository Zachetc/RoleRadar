"""Train and export the RoleRadar opportunity scoring model."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_FILE = Path("job_market_data.csv")
MODEL_PATH = Path("FlaskAPI/models/model_file.p")
RANDOM_STATE = 42

FEATURES = [
    "Rating", "Size", "Type of ownership", "Industry", "Sector", "Revenue", "num_comp",
    "job_state", "same_state", "age", "python_yn", "sql_yn", "excel_yn", "tableau_yn",
    "powerbi_yn", "aws_yn", "spark_yn", "snowflake_yn", "dbt_yn", "airflow_yn",
    "docker_yn", "kubernetes_yn", "machine_learning_yn", "statistics_yn", "genai_yn",
    "experimentation_yn", "etl_yn", "remote_yn", "hybrid_yn", "entry_level_yn",
    "senior_language_yn", "bachelors_required_yn", "masters_plus_yn", "stakeholder_yn",
    "leadership_yn", "streaming_yn", "analytics_delivery_yn", "years_experience", "desc_len",
    "skill_count", "core_stack_score", "modern_stack_score", "job_simp", "seniority",
    "company_size_score", "revenue_score", "remote_flex_score", "entry_access_score",
    "collaboration_score", "posting_quality_score"
]


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df["opportunity_score"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_estimators=400,
        max_depth=22,
        min_samples_leaf=2,
        n_jobs=1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    artifact = {
        "model": pipeline,
        "feature_columns": FEATURES,
        "metrics": {
            "mae": round(float(mean_absolute_error(y_test, preds)), 3),
            "r2": round(float(r2_score(y_test, preds)), 3),
        },
        "target": "opportunity_score",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "project": "roleradar",
        "version": "3.0.0",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print("Model: RandomForestRegressor(n_estimators=400, max_depth=22, min_samples_leaf=2)")
    print("Metrics:", artifact["metrics"])
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
