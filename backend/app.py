"""
MedPredict backend -- FastAPI service wrapping:
  - the Random Forest symptom -> condition classifier (with SHAP-based
    per-prediction explanations, not just a bare label),
  - the free-text symptom matcher (embedding-based entity linking), and
  - a haversine-distance + specialty-aware hospital locator.

The JavaFX desktop client is a thin UI over this API -- all the ML/NLP logic
lives here so it can be tested, versioned, and reused independently of the
desktop client (e.g. a future web or mobile client could reuse it unchanged).

Run: uvicorn app:app --reload --port 8000
"""
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ML_DIR = Path(__file__).parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
from symptom_matcher import SymptomMatcher  # noqa: E402
from disease_specialty import specialty_for  # noqa: E402

MODELS_DIR = ML_DIR / "models"
BACKEND_DIR = Path(__file__).parent

app = FastAPI(title="MedPredict API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model + metadata once at startup ----------------------------------
clf = joblib.load(MODELS_DIR / "model.joblib")
symptoms: List[str] = json.loads((MODELS_DIR / "symptoms.json").read_text())
symptom_phrases: List[str] = json.loads((MODELS_DIR / "symptom_phrases.json").read_text())
metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
explainer = shap.TreeExplainer(clf)
matcher = SymptomMatcher()
hospitals = json.loads((BACKEND_DIR / "hospitals.json").read_text())

CLASSES = clf.classes_.tolist()
PHRASE_BY_SYMPTOM = dict(zip(symptoms, symptom_phrases))


# --- Schemas -----------------------------------------------------------------
class MatchRequest(BaseModel):
    text: str


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., description="canonical symptom keys, e.g. ['fatigue', 'high_fever']")
    top_k: int = 3


class HospitalsRequest(BaseModel):
    lat: float
    lon: float
    disease: Optional[str] = None
    limit: int = 5


# --- Helpers -----------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_feature_row(present_symptoms: List[str]) -> pd.DataFrame:
    unknown = [s for s in present_symptoms if s not in symptoms]
    if unknown:
        raise HTTPException(400, f"Unknown symptom key(s): {unknown}")
    row = {s: (1 if s in present_symptoms else 0) for s in symptoms}
    return pd.DataFrame([row], columns=symptoms)


def explain_prediction(x_row: pd.DataFrame, class_index: int, present_symptoms: List[str], top_n=5):
    sv = explainer.shap_values(x_row)  # shape (1, n_features, n_classes)
    contributions = sv[0, :, class_index]
    present_idx = [symptoms.index(s) for s in present_symptoms]
    ranked = sorted(present_idx, key=lambda i: contributions[i], reverse=True)[:top_n]
    return [
        {
            "symptom": symptoms[i],
            "phrase": PHRASE_BY_SYMPTOM[symptoms[i]],
            "shap_contribution": round(float(contributions[i]), 4),
        }
        for i in ranked
    ]


# --- Routes ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_symptoms": len(symptoms),
        "n_diseases": len(CLASSES),
        "model_metrics": {
            "held_out_test_accuracy": metrics["held_out_test_accuracy"],
            "held_out_test_f1_macro": metrics["held_out_test_f1_macro"],
        },
    }


@app.get("/symptoms")
def list_symptoms():
    """All canonical symptoms, for a checklist-style UI."""
    return [{"key": s, "phrase": PHRASE_BY_SYMPTOM[s]} for s in symptoms]


@app.post("/match-symptoms")
def match_symptoms(req: MatchRequest):
    """Free-text -> canonical symptoms via embedding similarity (the NLP intake path)."""
    result = matcher.match(req.text)
    return {
        "matched": [
            {"segment": m.segment, "symptom": m.symptom, "phrase": m.phrase, "similarity": m.similarity}
            for m in result.matched
        ],
        "unmatched_segments": result.unmatched_segments,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.symptoms:
        raise HTTPException(400, "Provide at least one symptom.")

    x_row = build_feature_row(req.symptoms)
    proba = clf.predict_proba(x_row)[0]
    top_idx = np.argsort(proba)[::-1][: req.top_k]

    results = []
    for i in top_idx:
        i = int(i)
        results.append({
            "disease": clf.classes_[i],
            "confidence": round(float(proba[i]), 4),
            "recommended_specialty": specialty_for(clf.classes_[i]),
            "explanation": explain_prediction(x_row, i, req.symptoms),
        })

    return {"predictions": results}


@app.post("/hospitals")
def nearest_hospitals(req: HospitalsRequest):
    wanted_specialty = specialty_for(req.disease) if req.disease else None

    scored = []
    for h in hospitals:
        dist = haversine_km(req.lat, req.lon, h["lat"], h["lon"])
        has_specialty = wanted_specialty is not None and wanted_specialty in h["specialties"]
        scored.append({
            "name": h["name"],
            "lat": h["lat"],
            "lon": h["lon"],
            "specialties": h["specialties"],
            "distance_km": round(dist, 2),
            "matches_recommended_specialty": has_specialty,
        })

    # Relevant-specialty hospitals first, then by distance within each group --
    # a plain nearest-only sort would happily send a UTI patient to the
    # oncology-only hospital next door instead of the urology unit 2km further.
    scored.sort(key=lambda h: (not h["matches_recommended_specialty"], h["distance_km"]))

    return {
        "recommended_specialty": wanted_specialty,
        "hospitals": scored[: req.limit],
    }
