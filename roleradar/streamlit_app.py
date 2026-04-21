from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from job_market_intelligence import (
    build_input_record,
    clean_jobs_dataframe,
    detected_skill_labels,
    explain_posting,
    opportunity_band,
    summarize_posting,
    top_signal_drivers,
)

st.set_page_config(page_title="RoleRadar", layout="wide")

DATA_FILE = Path("job_market_data.csv")
MODEL_PATH = Path("FlaskAPI/models/model_file.p")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_artifact() -> dict:
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Market overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postings", f"{len(df):,}")
    c2.metric("Avg opportunity", f"{df['opportunity_score'].mean():.1f}")
    c3.metric("High-opportunity share", f"{(df['opportunity_score'] >= 80).mean()*100:.1f}%")
    c4.metric("Avg skills detected", f"{df['skill_count'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Top role families by average opportunity**")
        fam = df.groupby('job_simp', dropna=False)['opportunity_score'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fam.sort_values().plot(kind='barh', ax=ax)
        ax.set_xlabel('Average opportunity score')
        ax.set_ylabel('Job family')
        st.pyplot(fig)
    with right:
        st.markdown("**Most common in-demand skills**")
        skill_cols = [
            'python_yn','sql_yn','excel_yn','tableau_yn','powerbi_yn','aws_yn','spark_yn','snowflake_yn',
            'dbt_yn','airflow_yn','docker_yn','kubernetes_yn','machine_learning_yn','statistics_yn','genai_yn'
        ]
        skill_freq = (df[skill_cols].mean() * 100).sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        skill_freq.sort_values().plot(kind='barh', ax=ax2)
        ax2.set_xlabel('Percent of postings')
        ax2.set_ylabel('Signal')
        st.pyplot(fig2)

    lower = st.slider("Minimum opportunity score", 0, 100, 65)
    filtered = df[df['opportunity_score'] >= lower].sort_values('opportunity_score', ascending=False)
    st.markdown("**Top postings after filter**")
    st.dataframe(
        filtered[[
            'Job Title', 'company_txt', 'Location', 'job_simp', 'opportunity_score', 'opportunity_band', 'skill_count'
        ]].head(25),
        use_container_width=True,
    )


def render_scoring() -> None:
    st.subheader("Score a posting")
    with st.form("score_posting"):
        job_title = st.text_input("Job title", "Product Analyst")
        company_name = st.text_input("Company", "RoleRadar Demo Co")
        location = st.text_input("Location", "New York, NY")
        headquarters = st.text_input("Headquarters", "New York, NY")
        rating = st.number_input("Rating", min_value=-1.0, max_value=5.0, value=4.1, step=0.1)
        founded = st.number_input("Founded year", min_value=-1, max_value=2026, value=2017)
        size = st.selectbox("Company size", [
            '-1', '1 to 50 employees', '51 to 200 employees', '201 to 500 employees',
            '501 to 1000 employees', '1001 to 5000 employees', '5001 to 10000 employees', '10000+ employees'
        ], index=4)
        ownership = st.text_input("Ownership type", "Company - Private")
        industry = st.text_input("Industry", "Internet & Web Services")
        sector = st.text_input("Sector", "Information Technology")
        revenue = st.text_input("Revenue", "$100 to $500 million (USD)")
        description = st.text_area(
            "Job description",
            "We are hiring a product analyst with SQL, Python, Tableau, experimentation, KPI dashboards, and cross-functional stakeholder experience. Hybrid in New York. 2+ years preferred. Exposure to dbt, Snowflake, and LLM-enabled analytics is a plus.",
            height=220,
        )
        submitted = st.form_submit_button("Score posting")

    if submitted:
        artifact = load_artifact()
        payload = {
            "job_title": job_title,
            "job_description": description,
            "rating": rating,
            "company_name": company_name,
            "location": location,
            "headquarters": headquarters,
            "size": size,
            "founded": founded,
            "ownership_type": ownership,
            "industry": industry,
            "sector": sector,
            "revenue": revenue,
            "competitors": "-1",
        }
        raw_df = build_input_record(payload)
        engineered = clean_jobs_dataframe(raw_df)
        pred = float(artifact['model'].predict(engineered[artifact['feature_columns']])[0])
        row = engineered.iloc[0].copy()
        row['opportunity_score'] = pred
        row['opportunity_band'] = opportunity_band(pred)

        a, b, c = st.columns(3)
        a.metric("Opportunity score", f"{pred:.1f}")
        b.metric("Band", opportunity_band(pred).title())
        c.metric("Detected skills", int(row['skill_count']))

        st.info(summarize_posting(row))

        detail_left, detail_right = st.columns([1.2, 1])
        with detail_left:
            st.markdown("**Detected skills**")
            st.write(detected_skill_labels(row) or ["No major skills detected"])

            st.markdown("**Explanation**")
            for item in explain_posting(row):
                st.write(f"- {item}")
        with detail_right:
            st.markdown("**Top score drivers**")
            st.dataframe(pd.DataFrame(top_signal_drivers(row)), use_container_width=True)

        st.markdown("**Signal snapshot**")
        snapshot = pd.DataFrame({
            'signal': ['core_stack_score', 'modern_stack_score', 'posting_quality_score', 'entry_access_score', 'remote_flex_score'],
            'value': [
                row['core_stack_score'], row['modern_stack_score'], row['posting_quality_score'],
                row['entry_access_score'], row['remote_flex_score']
            ]
        })
        st.dataframe(snapshot, use_container_width=True)


def render_methodology(df: pd.DataFrame) -> None:
    st.subheader("Methodology")
    st.markdown(
        "RoleRadar combines rule-based text extraction with a supervised model. The target is an engineered **opportunity score** built from accessibility, stack relevance, flexibility, company context, and posting clarity."
    )
    st.markdown("**Signal families**")
    st.write({
        "Technical stack": ["Python", "SQL", "Tableau", "Power BI", "Snowflake", "dbt", "Airflow"],
        "Market signals": ["GenAI", "Streaming", "Experimentation", "Analytics delivery"],
        "Accessibility": ["Years of experience", "Entry-level language", "Degree language"],
        "Business context": ["Stakeholder exposure", "Leadership language", "Company size", "Revenue"],
    })
    st.markdown("**Opportunity band distribution**")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    df['opportunity_band'].value_counts().reindex(['low','moderate','strong','high']).plot(kind='bar', ax=ax)
    ax.set_xlabel('Opportunity band')
    ax.set_ylabel('Number of postings')
    st.pyplot(fig)


def main() -> None:
    st.title("RoleRadar")
    st.caption("Explainable job market intelligence for ranking and understanding job postings")
    df = load_data()
    tabs = st.tabs(["Dashboard", "Posting scorer", "Methodology"])
    with tabs[0]:
        render_overview(df)
    with tabs[1]:
        render_scoring()
    with tabs[2]:
        render_methodology(df)


if __name__ == '__main__':
    main()
