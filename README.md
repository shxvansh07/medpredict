# MedPredict — ML Symptom Checker & Hospital Locator

A symptom-to-condition predictor with a JavaFX desktop client, rebuilt around three ideas most
classroom symptom-checkers skip:

1. **You describe symptoms in plain English**, not by hunting through a 130-item checkbox list.
   A sentence-embedding matcher links free text ("it burns when I pee and I'm exhausted") to the
   right canonical symptoms, with a similarity score you can inspect.
2. **Every prediction explains itself.** The model doesn't just output "Urinary tract infection" —
   it ships a SHAP-based breakdown of which of *your* reported symptoms actually drove that specific
   prediction, computed per-request with `shap.TreeExplainer`, not a generic global importance chart.
3. **The hospital locator is specialty-aware, not just nearest-first.** It ranks hospitals by
   whether their department matches the predicted condition, then by distance — so a UTI doesn't
   route you to the oncology wing next door over the urology unit two kilometers further out.

## Architecture

```
┌─────────────────────┐        HTTP/JSON        ┌──────────────────────────┐
│   JavaFX Desktop     │ ───────────────────────▶│   FastAPI Backend        │
│   (Symptom Checker + │◀─────────────────────── │   /symptoms              │
│    Hospital Locator) │                          │   /match-symptoms        │
└─────────────────────┘                          │   /predict               │
                                                    │   /hospitals             │
                                                    └───────────┬──────────────┘
                                                                │
                                        ┌───────────────────────┼────────────────────────┐
                                        ▼                       ▼                        ▼
                              Random Forest classifier   SHAP TreeExplainer     Sentence-Transformer
                              (132 symptoms → 41          (per-prediction         symptom matcher
                               conditions)                 explanation)           (free-text intake)
```

The desktop client is a thin UI: all ML/NLP logic lives in the backend, so it can be tested,
versioned, and reused independently of the client (a future web or mobile front end could reuse it
unchanged).

## Why a separate ML microservice instead of embedding the model in Java?

The original version of this project ran everything inside the Java process. Splitting the ML/NLP
stack into its own FastAPI service is both more realistic (this is how most production ML products
are actually architected) and more practical: Python's ML/NLP ecosystem (scikit-learn, SHAP,
sentence-transformers) has no equivalent maturity on the JVM, and this way the model can be
retrained, redeployed, or swapped out without touching or recompiling the desktop client.

## Project layout

```
medpredict/
├── data/                    training_data.csv, test_data.csv (symptom → disease dataset)
├── ml/
│   ├── train.py             trains the Random Forest, saves model + metrics + SHAP feature ranking
│   ├── symptom_phrases.py   canonical symptom key -> natural-language phrase
│   ├── build_symptom_embeddings.py   precomputes sentence embeddings for every symptom phrase
│   └── symptom_matcher.py   free-text -> canonical symptom(s) via cosine similarity
├── backend/
│   ├── app.py                FastAPI app: /symptoms, /match-symptoms, /predict, /hospitals
│   ├── disease_specialty.py  disease -> recommended hospital specialty mapping
│   └── hospitals.json        sample hospital dataset (Bengaluru) with specialties + coordinates
└── desktop/                  JavaFX (Maven) client: Symptom Checker + Hospital Locator tabs
```

## Setup

### 1. Train the model (run once)

```bash
cd ml
pip install -r requirements.txt
python train.py                      # trains the Random Forest, saves models/
python build_symptom_embeddings.py   # precomputes symptom-phrase embeddings
```

### 2. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --port 8000
```

### 3. Run the desktop client

```bash
cd desktop
mvn javafx:run
```

The client expects the backend at `http://127.0.0.1:8000` (see `ApiClient` in `MedPredictApp.java`).

## The dataset, honestly

`data/training_data.csv` is the widely-used 132-symptom / 41-disease dataset ([source](https://github.com/anujdutt9/Disease-Prediction-from-Symptoms)). Each disease maps to a fixed
symptom combination repeated across rows, so 5-fold cross-validation on it scores a perfect
F1 = 1.0 — that's a ceiling effect of a clean, synthetic dataset, not evidence the model
generalizes to messy real-world symptom reporting. The held-out `test_data.csv` (one row per
disease, ~97.6% accuracy) is a slightly more honest signal, but still drawn from the same
distribution. Treat this as a solid systems/ML-pipeline project, not a validated diagnostic tool.

## Stack

- **ML:** Python, scikit-learn (Random Forest), SHAP (explainability)
- **NLP:** sentence-transformers (`all-MiniLM-L6-v2`) for free-text symptom matching
- **Backend:** FastAPI, Pydantic
- **Desktop:** Java, JavaFX (Maven), `java.net.http.HttpClient`, org.json
- **Geolocation:** haversine distance + specialty-aware ranking over a bundled hospital dataset

Built as a from-scratch rebuild of an earlier academic symptom-checker project, extended with an
NLP intake path, explainable predictions, and specialty-aware hospital ranking.
