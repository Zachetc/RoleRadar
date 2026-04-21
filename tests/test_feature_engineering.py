from job_market_intelligence import (
    build_input_record,
    clean_jobs_dataframe,
    opportunity_band,
    top_signal_drivers,
)


def test_clean_jobs_dataframe_extracts_core_signals():
    payload = {
        "job_title": "Junior Data Analyst",
        "job_description": "Entry level analyst role using SQL, Python, Tableau, dashboards, and stakeholder communication. Hybrid in NYC. 1 year of experience preferred.",
        "rating": 4.1,
        "company_name": "Demo Co",
        "location": "New York, NY",
        "headquarters": "New York, NY",
        "size": "1001 to 5000 employees",
        "founded": 2018,
        "ownership_type": "Company - Private",
        "industry": "Internet & Web Services",
        "sector": "Information Technology",
        "revenue": "$100 to $500 million (USD)",
        "competitors": "-1",
    }

    engineered = clean_jobs_dataframe(build_input_record(payload))
    row = engineered.iloc[0]

    assert row["sql_yn"] == 1
    assert row["python_yn"] == 1
    assert row["tableau_yn"] == 1
    assert row["hybrid_yn"] == 1
    assert row["stakeholder_yn"] == 1
    assert row["years_experience"] == 1
    assert row["job_simp"] in {"data analyst", "other"}


def test_opportunity_band_mapping():
    assert opportunity_band(30) == "low"
    assert opportunity_band(55) == "moderate"
    assert opportunity_band(70) == "strong"
    assert opportunity_band(85) == "high"


def test_top_signal_drivers_returns_ranked_items():
    payload = {
        "job_title": "Analytics Engineer",
        "job_description": "Build SQL, Python, dbt, Snowflake, Airflow, dashboards, and partner with stakeholders. Hybrid. 2 years experience.",
        "rating": 4.3,
        "company_name": "Demo Co",
        "location": "New York, NY",
        "headquarters": "New York, NY",
        "size": "1001 to 5000 employees",
        "founded": 2018,
        "ownership_type": "Company - Private",
        "industry": "Internet & Web Services",
        "sector": "Information Technology",
        "revenue": "$100 to $500 million (USD)",
        "competitors": "-1",
    }
    engineered = clean_jobs_dataframe(build_input_record(payload))
    drivers = top_signal_drivers(engineered.iloc[0])
    assert len(drivers) > 0
    assert drivers[0]["impact"] >= drivers[-1]["impact"]
