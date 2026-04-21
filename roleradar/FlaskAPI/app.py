from __future__ import annotations

import pickle
import sys
from pathlib import Path

import flask
from flask import Flask, request

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from job_market_intelligence import (
    build_input_record,
    clean_jobs_dataframe,
    detected_skill_labels,
    explain_posting,
    opportunity_band,
    summarize_posting,
    top_signal_drivers,
)

MODEL_PATH = CURRENT_DIR / "models" / "model_file.p"
app = Flask(__name__)


def load_artifact() -> dict:
    with open(MODEL_PATH, "rb") as pickled:
        return pickle.load(pickled)


def _score_payload(payload: dict, artifact: dict) -> dict:
    raw_df = build_input_record(payload)
    engineered = clean_jobs_dataframe(raw_df)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    prediction = float(model.predict(engineered[feature_columns])[0])
    row = engineered.iloc[0].copy()
    row["opportunity_score"] = prediction
    row["opportunity_band"] = opportunity_band(prediction)

    return {
        "predicted_opportunity_score": round(prediction, 1),
        "opportunity_band": opportunity_band(prediction),
        "market_summary": {
            "job_family": row["job_simp"],
            "seniority": row["seniority"],
            "skill_count": int(row["skill_count"]),
            "years_experience": int(row["years_experience"]),
            "remote": bool(row["remote_yn"]),
            "hybrid": bool(row["hybrid_yn"]),
            "high_opportunity_role": bool(prediction >= 65),
        },
        "detected_skills": detected_skill_labels(row),
        "signal_snapshot": {
            "core_stack_score": float(row["core_stack_score"]),
            "modern_stack_score": float(row["modern_stack_score"]),
            "posting_quality_score": float(round(row["posting_quality_score"], 2)),
            "entry_access_score": float(row["entry_access_score"]),
            "remote_flex_score": float(row["remote_flex_score"]),
        },
        "top_drivers": top_signal_drivers(row),
        "summary": summarize_posting(row),
        "explanations": explain_posting(row),
    }


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "project": "roleradar"}, 200


@app.route("/metadata", methods=["GET"])
def metadata():
    artifact = load_artifact()
    return {
        "project": artifact.get("project", "roleradar"),
        "version": artifact.get("version", "unknown"),
        "target": artifact.get("target", "opportunity_score"),
        "trained_at_utc": artifact.get("trained_at_utc"),
        "metrics": artifact.get("metrics", {}),
        "feature_count": len(artifact.get("feature_columns", [])),
    }, 200


@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.get_json(force=True) or {}
    payload = request_json.get("input", request_json)
    artifact = load_artifact()
    response = _score_payload(payload, artifact)
    response["project"] = "roleradar"
    response["model_metrics"] = artifact.get("metrics", {})
    return flask.jsonify(response), 200


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    request_json = request.get_json(force=True) or {}
    inputs = request_json.get("inputs", [])
    if not isinstance(inputs, list) or not inputs:
        return flask.jsonify({"error": "Provide a non-empty 'inputs' list."}), 400

    artifact = load_artifact()
    results = []
    for idx, payload in enumerate(inputs):
        scored = _score_payload(payload, artifact)
        scored["row_id"] = idx
        results.append(scored)

    return flask.jsonify({
        "project": "roleradar",
        "count": len(results),
        "results": results,
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
