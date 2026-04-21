"""Collect job postings for the job-market intelligence platform.

This keeps the original scraper-driven workflow, but broadens collection from a
single salary use case to a multi-role market snapshot across analytics,
engineering, BI, and AI-adjacent jobs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import glassdoor_scraper as gs

SEARCH_TERMS = [
    "data analyst",
    "business analyst",
    "business intelligence analyst",
    "analytics engineer",
    "data engineer",
    "machine learning engineer",
    "product analyst",
    "decision scientist",
]

CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "chromedriver")
NUM_JOBS_PER_TERM = int(os.getenv("NUM_JOBS_PER_TERM", "120"))
VERBOSE = os.getenv("SCRAPER_VERBOSE", "false").lower() == "true"
SLEEP_TIME = int(os.getenv("SCRAPER_SLEEP_SECONDS", "10"))
OUTPUT_FILE = Path("job_market_postings.csv")


def main() -> None:
    frames = []
    for term in SEARCH_TERMS:
        df = gs.get_jobs(term, NUM_JOBS_PER_TERM, VERBOSE, CHROMEDRIVER_PATH, SLEEP_TIME)
        df["search_term"] = term
        frames.append(df)

    output = pd.concat(frames, ignore_index=True)
    output = output.drop_duplicates(subset=["Job Title", "Company Name", "Location"])
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} with shape {output.shape}")


if __name__ == "__main__":
    main()
