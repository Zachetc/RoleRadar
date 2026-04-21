from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from job_market_intelligence import clean_jobs_dataframe, detected_skill_labels, explain_posting

DATA_FILE = Path("glassdoor_jobs.csv")
MODEL_PATH = Path("FlaskAPI/models/model_file.p")
OUTPUT_FILE = Path("ranked_job_postings.csv")


def main() -> None:
    with open(MODEL_PATH, "rb") as fh:
        artifact = pickle.load(fh)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    raw_df = pd.read_csv(DATA_FILE)
    engineered = clean_jobs_dataframe(raw_df)
    engineered["predicted_opportunity_score"] = model.predict(engineered[feature_columns]).round(1)
    engineered["opportunity_band"] = engineered["predicted_opportunity_score"].apply(
        lambda x: "high" if x >= 80 else "strong" if x >= 65 else "moderate" if x >= 45 else "low"
    )
    engineered["detected_skills"] = engineered.apply(detected_skill_labels, axis=1)
    engineered["explanation_preview"] = engineered.apply(lambda row: " | ".join(explain_posting(row)[:3]), axis=1)

    export_cols = [
        "Job Title", "company_txt", "Location", "job_simp", "seniority",
        "predicted_opportunity_score", "opportunity_band", "years_experience",
        "remote_yn", "hybrid_yn", "skill_count", "detected_skills", "explanation_preview"
    ]
    available = [c for c in export_cols if c in engineered.columns]
    ranked = engineered.sort_values("predicted_opportunity_score", ascending=False)[available]
    ranked.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved ranked postings to {OUTPUT_FILE.resolve()}")
    print(ranked.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
