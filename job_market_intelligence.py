from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
UNKNOWN_TOKEN = "-1"


SKILL_PATTERNS: Dict[str, Sequence[str]] = {
    "python_yn": [r"\bpython\b"],
    "sql_yn": [r"\bsql\b", r"postgresql", r"mysql", r"sql server", r"redshift", r"bigquery"],
    "excel_yn": [r"\bexcel\b", r"spreadsheets?"],
    "tableau_yn": [r"\btableau\b"],
    "powerbi_yn": [r"power\s*bi", r"powerbi"],
    "aws_yn": [r"\baws\b", r"amazon web services"],
    "spark_yn": [r"\bspark\b", r"pyspark"],
    "snowflake_yn": [r"\bsnowflake\b"],
    "dbt_yn": [r"\bdbt\b"],
    "airflow_yn": [r"\bairflow\b"],
    "docker_yn": [r"\bdocker\b", r"containeri[sz]ation"],
    "kubernetes_yn": [r"\bkubernetes\b", r"\bk8s\b"],
    "machine_learning_yn": [r"machine learning", r"predictive modeling", r"supervised learning", r"mlops"],
    "statistics_yn": [r"statistics", r"statistical", r"hypothesis testing", r"regression", r"experimentation"],
    "genai_yn": [r"generative ai", r"large language model", r"\bllm\b", r"prompt engineering"],
    "experimentation_yn": [r"a/?b testing", r"experimentation", r"causal inference"],
    "etl_yn": [r"\betl\b", r"data pipeline", r"data orchestration"],
}

REMOTE_PATTERNS = [r"\bremote\b", r"work from home", r"distributed team"]
HYBRID_PATTERNS = [r"\bhybrid\b", r"in-office .* days", r"onsite .* days"]
ENTRY_PATTERNS = [r"entry level", r"junior", r"associate", r"new grad", r"0[- ]?2 years", r"early career"]
SENIOR_PATTERNS = [r"senior", r"staff", r"lead", r"principal", r"director", r"manager"]
MASTERS_PATTERNS = [r"master'?s", r"\bms\b", r"\bmba\b", r"\bphd\b", r"doctorate"]
BACHELORS_PATTERNS = [r"bachelor'?s", r"\bbs\b", r"\bba\b"]
STAKEHOLDER_PATTERNS = [r"stakeholder", r"cross-functional", r"executive", r"business partner", r"client"]
LEADERSHIP_PATTERNS = [r"leadership", r"mentor", r"manage", r"roadmap", r"own the strategy"]
STREAMING_PATTERNS = [r"real[- ]time", r"streaming", r"kafka", r"event[- ]driven"]
ANALYTICS_PATTERNS = [r"dashboard", r"reporting", r"kpi", r"metric", r"insight"]

EXPERIENCE_RE = re.compile(r"(\d+)\+?\s*(?:-|to)?\s*(\d+)?\s+years?", re.IGNORECASE)

JOB_SIMPLIFICATIONS = {
    "data scientist": ["data scientist"],
    "data analyst": ["data analyst", "analytics analyst"],
    "data engineer": ["data engineer", "etl developer", "analytics engineer", "warehouse engineer"],
    "machine learning engineer": ["machine learning engineer", "ml engineer", "mlops engineer"],
    "business intelligence": ["business intelligence", "bi analyst", "bi developer", "analytics consultant"],
    "product analyst": ["product analyst", "growth analyst"],
    "research scientist": ["research scientist", "applied scientist"],
    "manager": ["manager", "head of", "director", "lead"],
}

SENIORITY_MAP = {
    "senior": ["senior", "sr", "staff", "principal", "lead"],
    "junior": ["junior", "jr", "entry", "associate", "new grad"],
    "manager": ["manager", "director", "head"],
}

SIZE_SCORE = {
    "1 to 50 employees": 4,
    "51 to 200 employees": 7,
    "201 to 500 employees": 9,
    "501 to 1000 employees": 11,
    "1001 to 5000 employees": 14,
    "5001 to 10000 employees": 12,
    "10000+ employees": 10,
    UNKNOWN_TOKEN: 6,
    np.nan: 6,
}

REVENUE_SCORE = {
    "$1 to $5 million (usd)": 4,
    "$5 to $10 million (usd)": 5,
    "$10 to $25 million (usd)": 6,
    "$25 to $50 million (usd)": 7,
    "$50 to $100 million (usd)": 8,
    "$100 to $500 million (usd)": 10,
    "$500 million to $1 billion (usd)": 12,
    "$1 to $2 billion (usd)": 14,
    "$2 to $5 billion (usd)": 15,
    "$5 to $10 billion (usd)": 14,
    "$10+ billion (usd)": 13,
    UNKNOWN_TOKEN: 6,
}

SKILL_LABELS = {
    "Python": "python_yn",
    "SQL": "sql_yn",
    "Excel": "excel_yn",
    "Tableau": "tableau_yn",
    "Power BI": "powerbi_yn",
    "AWS": "aws_yn",
    "Spark": "spark_yn",
    "Snowflake": "snowflake_yn",
    "dbt": "dbt_yn",
    "Airflow": "airflow_yn",
    "Docker": "docker_yn",
    "Kubernetes": "kubernetes_yn",
    "Machine Learning": "machine_learning_yn",
    "Statistics": "statistics_yn",
    "Generative AI": "genai_yn",
    "Experimentation": "experimentation_yn",
    "ETL": "etl_yn",
}


@dataclass(frozen=True)
class OpportunityBands:
    low: float = 45.0
    medium: float = 65.0
    high: float = 80.0


BANDS = OpportunityBands()


DRIVER_WEIGHTS = {
    "core_stack_score": 1.0,
    "modern_stack_score": 1.0,
    "posting_quality_score": 1.0,
    "entry_access_score": 1.0,
    "remote_flex_score": 1.0,
    "collaboration_score": 1.0,
    "company_size_score": 0.9,
    "revenue_score": 0.8,
    "rating_score": 0.8,
    "skill_count": 0.6,
    "years_experience": -2.0,
    "streaming_yn": 2.0,
    "genai_yn": 2.0,
}

DRIVER_DISPLAY = {
    "core_stack_score": "Core analytics stack",
    "modern_stack_score": "Modern data tooling",
    "posting_quality_score": "Posting quality",
    "entry_access_score": "Accessibility for early-career talent",
    "remote_flex_score": "Workplace flexibility",
    "collaboration_score": "Business exposure",
    "company_size_score": "Company scale",
    "revenue_score": "Revenue profile",
    "rating_score": "Company rating",
    "skill_count": "Breadth of skills requested",
    "years_experience": "Experience requirement",
    "streaming_yn": "Real-time / streaming signal",
    "genai_yn": "Generative AI signal",
}


def _contains_any(text: str, patterns: Iterable[str]) -> int:
    text = text or ""
    return int(any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns))


def extract_years_experience(text: str) -> int:
    if not text:
        return 0
    matches = EXPERIENCE_RE.findall(text)
    if not matches:
        return 0
    nums: List[int] = []
    for low, high in matches:
        nums.append(int(high or low))
    return min(nums) if nums else 0


def simplify_job_title(title: str) -> str:
    title = (title or "").lower()
    for label, patterns in JOB_SIMPLIFICATIONS.items():
        if any(pattern in title for pattern in patterns):
            return label
    return "other"


def extract_seniority(title: str) -> str:
    title = (title or "").lower()
    for label, patterns in SENIORITY_MAP.items():
        if any(pattern in title for pattern in patterns):
            return label
    return "mid"


def revenue_score(value: str) -> int:
    if pd.isna(value):
        return 6
    return REVENUE_SCORE.get(str(value).lower(), 6)


def size_score(value: str) -> int:
    if pd.isna(value):
        return 6
    return SIZE_SCORE.get(str(value), 6)


def normalize_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    return df


def opportunity_band(score: float) -> str:
    if score >= BANDS.high:
        return "high"
    if score >= BANDS.medium:
        return "strong"
    if score >= BANDS.low:
        return "moderate"
    return "low"


def clean_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    normalize_text_columns(
        df,
        [
            "Company Name",
            "Location",
            "Headquarters",
            "Job Title",
            "Job Description",
            "Type of ownership",
            "Industry",
            "Sector",
            "Revenue",
            "Competitors",
            "Size",
        ],
    )

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(-1)
    df["company_txt"] = df.apply(
        lambda row: row["Company Name"] if row["Rating"] < 0 else str(row["Company Name"]).rsplit("\n", 1)[0],
        axis=1,
    )
    df["job_state"] = df["Location"].apply(lambda x: x.split(",")[1].strip() if "," in x else "Unknown")
    df["same_state"] = df.apply(
        lambda row: int(str(row.get("Location", "")).strip() == str(row.get("Headquarters", "")).strip()),
        axis=1,
    )
    df["age"] = df["Founded"].apply(lambda x: 0 if pd.isna(x) or int(x) < 1 else CURRENT_YEAR - int(x))

    descriptions = df["Job Description"]
    for column, patterns in SKILL_PATTERNS.items():
        df[column] = descriptions.apply(lambda text: _contains_any(text, patterns))

    df["remote_yn"] = descriptions.apply(lambda text: _contains_any(text, REMOTE_PATTERNS))
    df["hybrid_yn"] = descriptions.apply(lambda text: _contains_any(text, HYBRID_PATTERNS))
    df["entry_level_yn"] = descriptions.apply(lambda text: _contains_any(text, ENTRY_PATTERNS))
    df["senior_language_yn"] = descriptions.apply(lambda text: _contains_any(text, SENIOR_PATTERNS))
    df["bachelors_required_yn"] = descriptions.apply(lambda text: _contains_any(text, BACHELORS_PATTERNS))
    df["masters_plus_yn"] = descriptions.apply(lambda text: _contains_any(text, MASTERS_PATTERNS))
    df["stakeholder_yn"] = descriptions.apply(lambda text: _contains_any(text, STAKEHOLDER_PATTERNS))
    df["leadership_yn"] = descriptions.apply(lambda text: _contains_any(text, LEADERSHIP_PATTERNS))
    df["streaming_yn"] = descriptions.apply(lambda text: _contains_any(text, STREAMING_PATTERNS))
    df["analytics_delivery_yn"] = descriptions.apply(lambda text: _contains_any(text, ANALYTICS_PATTERNS))

    df["years_experience"] = descriptions.apply(extract_years_experience)
    df["desc_len"] = descriptions.str.len()
    df["job_simp"] = df["Job Title"].apply(simplify_job_title)
    df["seniority"] = df["Job Title"].apply(extract_seniority)

    skill_cols = list(SKILL_PATTERNS.keys())
    df["skill_count"] = df[skill_cols].sum(axis=1)
    df["core_stack_score"] = (
        df[["python_yn", "sql_yn", "excel_yn", "tableau_yn", "powerbi_yn"]].sum(axis=1) * 3
    )
    df["modern_stack_score"] = (
        df[["snowflake_yn", "dbt_yn", "airflow_yn", "docker_yn", "kubernetes_yn", "genai_yn"]].sum(axis=1) * 4
    )
    df["company_size_score"] = df["Size"].apply(size_score)
    df["revenue_score"] = df["Revenue"].apply(revenue_score)
    df["rating_score"] = df["Rating"].apply(lambda x: 0 if x < 0 else round(float(x) * 4, 2))
    df["remote_flex_score"] = (df["remote_yn"] * 8) + (df["hybrid_yn"] * 4)
    df["entry_access_score"] = (
        (df["entry_level_yn"] * 8)
        + ((df["years_experience"] <= 2).astype(int) * 6)
        - (df["masters_plus_yn"] * 4)
        - (df["senior_language_yn"] * 4)
    )
    df["collaboration_score"] = (df["stakeholder_yn"] * 4) + (df["leadership_yn"] * 4) + (df["analytics_delivery_yn"] * 3)
    df["posting_quality_score"] = (
        (df["desc_len"].clip(upper=5000) / 5000 * 20)
        + (df["stakeholder_yn"] * 4)
        + (df["leadership_yn"] * 4)
        + (df["bachelors_required_yn"] * 2)
        + (df["analytics_delivery_yn"] * 2)
    )

    competitors = df["Competitors"].fillna(UNKNOWN_TOKEN).astype(str)
    df["num_comp"] = competitors.apply(lambda x: 0 if x == UNKNOWN_TOKEN else len([c for c in x.split(",") if c.strip()]))

    df["opportunity_score"] = (
        df["rating_score"]
        + (df["skill_count"] * 3.5)
        + df["core_stack_score"]
        + df["modern_stack_score"]
        + df["company_size_score"]
        + df["revenue_score"]
        + df["remote_flex_score"]
        + df["entry_access_score"]
        + df["collaboration_score"]
        + df["posting_quality_score"]
        + (df["streaming_yn"] * 2)
        - np.clip(df["years_experience"] - 3, 0, 10) * 2
    )
    df["opportunity_score"] = df["opportunity_score"].clip(lower=0, upper=100).round(1)
    df["opportunity_band"] = df["opportunity_score"].apply(opportunity_band)
    df["high_opportunity_role"] = (df["opportunity_score"] >= 65).astype(int)

    return df


def build_input_record(payload: Dict[str, object]) -> pd.DataFrame:
    fields = {
        "Job Title": payload.get("job_title", ""),
        "Job Description": payload.get("job_description", ""),
        "Rating": payload.get("rating", -1),
        "Company Name": payload.get("company_name", "Unknown Company"),
        "Location": payload.get("location", "Unknown, NA"),
        "Headquarters": payload.get("headquarters", "Unknown, NA"),
        "Size": payload.get("size", UNKNOWN_TOKEN),
        "Founded": payload.get("founded", -1),
        "Type of ownership": payload.get("ownership_type", UNKNOWN_TOKEN),
        "Industry": payload.get("industry", UNKNOWN_TOKEN),
        "Sector": payload.get("sector", UNKNOWN_TOKEN),
        "Revenue": payload.get("revenue", UNKNOWN_TOKEN),
        "Competitors": payload.get("competitors", UNKNOWN_TOKEN),
    }
    return pd.DataFrame([fields])


def detected_skill_labels(row: pd.Series) -> List[str]:
    return [label for label, col in SKILL_LABELS.items() if row.get(col, 0) == 1]


def summarize_posting(row: pd.Series) -> str:
    skills = detected_skill_labels(row)
    skill_phrase = ", ".join(skills[:4]) if skills else "core analytics skills"
    return (
        f"This {row.get('seniority', 'mid')}-level {row.get('job_simp', 'analytics')} posting shows demand for {skill_phrase} "
        f"and lands in the {row.get('opportunity_band', 'moderate')} opportunity band."
    )


def top_signal_drivers(row: pd.Series, limit: int = 5) -> List[Dict[str, object]]:
    contributions = []
    for key, weight in DRIVER_WEIGHTS.items():
        value = float(row.get(key, 0) or 0)
        impact = value * weight
        if abs(impact) < 1:
            continue
        direction = "positive" if impact >= 0 else "negative"
        contributions.append(
            {
                "signal": DRIVER_DISPLAY.get(key, key),
                "direction": direction,
                "impact": round(abs(impact), 1),
                "value": round(value, 1),
            }
        )
    contributions.sort(key=lambda item: item["impact"], reverse=True)
    return contributions[:limit]


def explain_posting(row: pd.Series) -> List[str]:
    explanations: List[str] = []
    if row.get("modern_stack_score", 0) >= 8:
        explanations.append("Posting references newer cloud or platform tooling, which strengthens market relevance.")
    if row.get("skill_count", 0) >= 6:
        explanations.append("Posting signals a broad analytics stack with several in-demand tools.")
    elif row.get("skill_count", 0) >= 3:
        explanations.append("Posting includes multiple relevant data skills, suggesting steady hiring demand.")

    if row.get("remote_yn", 0) == 1:
        explanations.append("Remote language improves flexibility and expands the reachable talent pool.")
    elif row.get("hybrid_yn", 0) == 1:
        explanations.append("Hybrid setup adds flexibility without being fully on-site.")

    if row.get("entry_level_yn", 0) == 1 or row.get("years_experience", 0) <= 2:
        explanations.append("Experience expectations look relatively accessible for early-career applicants.")
    elif row.get("years_experience", 0) >= 5:
        explanations.append("Higher experience requirements raise the barrier to entry.")

    if row.get("masters_plus_yn", 0) == 1:
        explanations.append("Advanced degree language may narrow the candidate pool.")
    if row.get("stakeholder_yn", 0) == 1:
        explanations.append("Cross-functional language suggests the role has business visibility beyond pure execution.")
    if row.get("Rating", -1) >= 4:
        explanations.append("Company rating is relatively strong, which can be a positive quality signal.")
    if row.get("desc_len", 0) >= 1800:
        explanations.append("Detailed job description suggests a more mature hiring process and clearer scope.")
    if row.get("genai_yn", 0) == 1:
        explanations.append("Generative AI language makes the posting feel more current and trend-aligned.")
    return explanations
