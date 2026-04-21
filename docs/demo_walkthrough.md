# Demo Walkthrough

## Fastest demo path
1. Create the virtual environment and install requirements.
2. Run `python model_building.py` to rebuild the model artifact.
3. Run `streamlit run streamlit_app.py`.
4. Open the **Posting scorer** tab.
5. Paste a realistic analytics role description.
6. Show the score, opportunity band, detected skills, explanation bullets, and top score drivers.

## Optional API demo
```bash
cd FlaskAPI
python app.py
```

Example request:
```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"job_title\":\"Analytics Engineer\",\"job_description\":\"Build dbt models, Airflow DAGs, SQL pipelines, dashboards, and support stakeholders. Hybrid in NYC. 2+ years required.\",\"rating\":4.2,\"company_name\":\"Demo Co\",\"location\":\"New York, NY\",\"headquarters\":\"New York, NY\",\"size\":\"1001 to 5000 employees\",\"founded\":2018,\"ownership_type\":\"Company - Private\",\"industry\":\"Internet & Web Services\",\"sector\":\"Information Technology\",\"revenue\":\"$100 to $500 million (USD)\",\"competitors\":\"-1\"}"
```

## What to point out live
- The score is explainable, not just a black box.
- The repo supports both single scoring and batch ranking.
- The project is framed as a market intelligence tool, not a toy prediction demo.
