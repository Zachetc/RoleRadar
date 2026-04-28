# RoleRadar

### Job Market Intelligence Platform for Opportunity Scoring & Hiring Signal Extraction

RoleRadar is an end-to-end machine learning platform that converts raw job postings into structured hiring intelligence.

Instead of predicting salaries, RoleRadar answers:

**Which jobs are actually worth applying to?**

It extracts required skills, detects seniority expectations, measures collaboration exposure, and produces an **Opportunity Score (0–100)** to help prioritize high-value analytics and data roles.

---

## Why RoleRadar Exists

Modern job searches involve hundreds of postings with vague expectations and inconsistent skill requirements.

RoleRadar transforms job descriptions into structured signals using:

* feature engineering pipelines
* NLP keyword extraction
* opportunity scoring models
* batch ranking workflows
* REST API delivery
* Streamlit analytics dashboard

The result is a system that ranks job postings by **market relevance, accessibility, and technical alignment**.

---

## Example Prediction Output

```json
{
  "opportunity_score": 76.3,
  "opportunity_band": "Strong",
  "role_family": "Data Analyst",
  "detected_skills": ["Python", "SQL", "Tableau"],
  "top_drivers": [
    "Strong analytics stack detected",
    "Collaboration signals present",
    "Modern tooling references"
  ]
}
```

---

## Architecture

Pipeline structure:

```
Scraper → Feature Engineering → Opportunity Model → Batch Ranking → API → Dashboard
```

See the architecture diagram:

```
assets/architecture.png
```

---

## Core Features

### Hiring Signal Extraction

Detects:

* Python
* SQL
* Tableau
* Power BI
* Snowflake
* dbt
* Airflow
* Spark
* AWS
* Docker
* Generative AI / LLM references
* experimentation / A/B testing signals
* stakeholder collaboration expectations
* dashboard ownership signals

---

### Opportunity Score Model

Outputs a score from **0–100** representing:

* stack strength
* tooling modernity
* collaboration exposure
* technical depth
* seniority expectations

Score bands:

| Score  | Meaning                   |
| ------ | ------------------------- |
| 80–100 | High-priority target role |
| 60–79  | Strong opportunity        |
| 40–59  | Moderate alignment        |
| <40    | Low-value posting         |

---

### Batch Ranking Engine

Score entire datasets instantly:

```
python batch_score.py
```

Outputs:

```
ranked_job_postings.csv
```

---

### REST API

Run locally:

```
cd FlaskAPI
python app.py
```

Example request:

```
POST /predict
```

Returns structured hiring intelligence JSON.

---

### Streamlit Dashboard

Launch:

```
streamlit run streamlit_app.py
```

Dashboard supports:

* single posting scoring
* dataset ranking
* methodology explanation
* hiring signal visualization

---

## Model Performance

Current model metrics:

| Metric | Value |
| ------ | ----- |
| R²     | 0.94  |
| MAE    | 3.06  |

Trained using engineered hiring-signal features extracted from job descriptions and company metadata.

---

## Repository Structure

```
RoleRadar/
│
├── FlaskAPI/
├── assets/
├── docs/
├── tests/
│
├── batch_score.py
├── data_collection.py
├── data_cleaning.py
├── job_market_intelligence.py
├── model_building.py
├── streamlit_app.py
```

---

## Technologies Used

Python
Pandas
Scikit-learn
Flask
Streamlit
Matplotlib
Selenium
Feature engineering pipelines
REST APIs

---

## Example Use Cases

RoleRadar supports:

* job search prioritization
* analytics skill-gap identification
* hiring trend analysis
* resume targeting strategy
* labor market intelligence exploration

---

## Planned Improvements

Future upgrades:

* SHAP feature attribution
* cloud deployment
* PostgreSQL dataset backend
* automated scraping scheduler
* hosted inference API

---

## Author

**Zachary Amachee**
CIS @ Baruch College
Analytics • Data Science • ML Engineering

GitHub:
https://github.com/Zachetc
