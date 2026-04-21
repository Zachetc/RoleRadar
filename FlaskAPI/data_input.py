SAMPLE_INPUT = {
    "input": {
        "job_title": "Product Analyst",
        "job_description": "We are hiring a product analyst with SQL, Python, Tableau, KPI dashboarding, experimentation, and cross-functional stakeholder experience. Hybrid in New York. 2+ years preferred. Snowflake and dbt exposure is a plus.",
        "rating": 4.2,
        "company_name": "RoleRadar Demo Co",
        "location": "New York, NY",
        "headquarters": "New York, NY",
        "size": "201 to 500 employees",
        "founded": 2018,
        "ownership_type": "Company - Private",
        "industry": "Internet & Web Services",
        "sector": "Information Technology",
        "revenue": "$100 to $500 million (USD)",
        "competitors": "-1"
    }
}

BATCH_INPUT = {
    "inputs": [
        SAMPLE_INPUT["input"],
        {
            "job_title": "Analytics Engineer",
            "job_description": "Analytics engineer role requiring SQL, dbt, Snowflake, Airflow, dashboard delivery, and stakeholder communication. Remote with 3+ years experience.",
            "rating": 4.0,
            "company_name": "Northstar Metrics",
            "location": "Boston, MA",
            "headquarters": "Boston, MA",
            "size": "501 to 1000 employees",
            "founded": 2016,
            "ownership_type": "Company - Private",
            "industry": "Enterprise Software",
            "sector": "Information Technology",
            "revenue": "$100 to $500 million (USD)",
            "competitors": "-1"
        }
    ]
}
