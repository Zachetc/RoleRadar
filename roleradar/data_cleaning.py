"""Build a market-intelligence dataset from raw job postings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from job_market_intelligence import clean_jobs_dataframe

RAW_FILE = Path("glassdoor_jobs.csv")
OUTPUT_FILE = Path("job_market_data.csv")


def main() -> None:
    df = pd.read_csv(RAW_FILE)
    cleaned = clean_jobs_dataframe(df)
    cleaned = cleaned.sort_values(["opportunity_score", "Rating"], ascending=[False, False]).reset_index(drop=True)
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} with shape {cleaned.shape}")
    print("Opportunity score summary:")
    print(cleaned["opportunity_score"].describe().round(2))


if __name__ == "__main__":
    main()
