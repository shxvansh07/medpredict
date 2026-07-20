# MedPredict — Study & Interview Prep

Everything below was verified against the actual code on 2026-07-21 (not the commit
message or README claims taken at face value). Where the implementation is simpler
or different than it sounds, that's called out explicitly.

---

## 1. The 60–90 second pitch

"MedPredict is a three-tier symptom-checker and hospital-locator I built to go beyond
the typical classroom version, which is usually a single script with a hardcoded
checkbox list. I split it into a JavaFX desktop client, a FastAPI backend, and a
Python ML pipeline that talk over plain HTTP/JSON.

The ML side trains a Random Forest on a public 132-symptom/41-disease dataset, and
every prediction ships a *per-request* SHAP explanation — using `shap.TreeExplainer`
— so instead of just 'Urinary Tract Infection' the user sees which of their specific
reported symptoms actually pushed that prediction, with numeric contribution scores.

On top of that there's an NLP intake path: instead of hunting through 132 jargon-heavy
checkboxes, a user can type 'it burns when I pee and I'm exhausted' and a
sentence-embedding matcher (`all-MiniLM-L6-v2` via sentence-transformers) segments
that text and cosine-matches each segment to the closest canonical symptom, showing
its similarity score so the match is inspectable, not a black box.

The hospital locator is specialty-aware rather than nearest-first: it ranks hospitals
by whether their listed specialty matches the predicted condition's recommended
specialty, then by haversine distance — so a UTI patient isn't routed to an
oncology-only hospital next door over a urology unit two kilometers further out.

The reason I split it into three tiers — Java desktop, Python API, Python ML — rather
than one Java app is that Python's ML/NLP ecosystem (scikit-learn, SHAP,
sentence-transformers) has no real JVM equivalent, and this mirrors how ML products
are actually shipped in production: a thin client hitting a versioned model-serving
API that can be redeployed independently."

---

## 2. Architecture walkthrough

```
┌──────────────────────┐   HTTP/JSON (java.net.http.HttpClient)   ┌───────────────────────────┐
│ JavaFX Desktop Client │ ─────────────────────────────────────▶ │ FastAPI Backend (app.py)  │
│  SymptomCheckerPane   │ ◀───────────────────────────────────── │ GET  /health              │
│  HospitalLocatorPane  │                                          │ GET  /symptoms            │
└──────────────────────┘                                          │ POST /match-symptoms      │
                                                                    │ POST /predict             │
                                                                    │ POST /hospitals           │
                                                                    └─────────────┬─────────────┘
                                                                                  │ imports at startup
                                          ┌───────────────────────────────────────┼───────────────────────────┐
                                          ▼                                       ▼                           ▼
                              ml/symptom_matcher.py                 backend/disease_specialty.py    ml/models/*.joblib,*.json,*.npy
                              (SentenceTransformer +                 (disease -> specialty lookup      (RandomForestClassifier,
                               cosine similarity)                    dict, case-insensitive)            symptom list, SHAP-ready)
```

**Java desktop tier** (`desktop/src/main/java/com/medpredict/`):
- `MedPredictApp.java` — entry point, wires `ApiClient` to `http://127.0.0.1:8000`,
  does a `/health` check on launch and shows a warning `Alert` (not a hard failure)
  if the backend isn't reachable.
- `ApiClient.java` — thin `java.net.http.HttpClient` wrapper. Notably forces
  **HTTP/1.1** explicitly (`HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)`)
  with a comment explaining why: the JDK client's default HTTP/2-upgrade attempt over
  plain `http://` can silently drop the POST body against a plain HTTP/1.1 server like
  uvicorn, surfacing as a confusing 422 "field required" even though the client did
  send a body. This is a real, non-obvious gotcha worth knowing cold for interviews.
- `SymptomCheckerPane.java` / `HospitalLocatorPane.java` — UI + a `Thread` per network
  call (`new Thread(() -> {...}, "predict").start()`), marshaling results back to the
  FX thread via `Platform.runLater(...)`. No executor/thread pool — a new raw thread
  per request, acceptable at this scale but worth naming as a simplification.
- `AppState.java` — trivial shared mutable state (`volatile String lastPredictedDisease`)
  so a prediction can prefill the hospital tab's "Condition" field.

**Python backend tier** (`backend/app.py`, FastAPI): loads the trained model, symptom
list, SHAP explainer, the symptom matcher, and `hospitals.json` once at import/startup
(module-level globals) — not per-request — so predictions are fast after the initial
model load. Five routes: `/health`, `/symptoms`, `/match-symptoms`, `/predict`,
`/hospitals`. Verified all five respond correctly end-to-end (see Section "What I
verified" below).

**ML pipeline** (`ml/`):
- `train.py` — trains `RandomForestClassifier(n_estimators=300, class_weight="balanced")`
  on `data/training_data.csv` (132 binary symptom columns → `prognosis` label), evaluates
  on the separate `data/test_data.csv`, and also runs 5-fold CV on the training split.
  Saves `model.joblib`, `symptoms.json`, `diseases.json`, `metrics.json`.
- `symptom_phrases.py` — hand-written natural-language phrase per jargon-heavy symptom
  key (e.g. `burning_micturition` → "burning sensation while urinating"), falling back
  to a cleaned column name otherwise.
- `build_symptom_embeddings.py` — embeds every symptom phrase once with
  `all-MiniLM-L6-v2` and saves `symptom_embeddings.npy`, so the matcher doesn't
  re-embed the anchor set on every API call.
- `symptom_matcher.py` — `SymptomMatcher.match(text)` splits free text on
  commas/"and"/"also"/"plus"/newlines/semicolons via regex, embeds each segment,
  computes cosine similarity against the precomputed symptom embeddings, and accepts
  the best match only above `MATCH_THRESHOLD = 0.42` (otherwise the segment is reported
  as "unmatched" rather than guessed).

---

## 3. Key concepts to explain in depth

**Symptom matching is cosine similarity over sentence embeddings, not a trained
classifier or NER model.** There is no fine-tuning here: `all-MiniLM-L6-v2` is used
off-the-shelf to embed both the 132 fixed "anchor" symptom phrases and the user's
free-text segments, and matching is `argmax` of the dot product (vectors are
L2-normalized via `normalize_embeddings=True`, so dot product = cosine similarity).
This is a deliberate simplicity/explainability trade-off: every match ships its
similarity score and the exact anchor phrase it was compared against, which a
black-box NER model wouldn't give you.

**The disease predictor is a Random Forest classifier**, not a neural net, over 132
binary symptom features → 41 disease classes. `class_weight="balanced"` is set (worth
knowing why: to avoid the model just favoring more frequent classes if the dataset
isn't perfectly balanced).

**SHAP explainability is real and per-request**, not decorative. `backend/app.py`
builds one `shap.TreeExplainer(clf)` at startup and calls `explainer.shap_values(x_row)`
inside `/predict` for the *specific* row of symptoms just submitted, then ranks only
the symptoms the user actually reported by their SHAP contribution to the predicted
class (`explain_prediction()` in app.py). This matches the commit message's claim —
confirmed by reading the code, not assumed from it. `train.py` also computes a
*global* SHAP importance ranking once at training time (`top_20_global_symptoms_by_shap`
in `metrics.json`), which is a separate, coarser artifact from the per-prediction
explanation the API actually serves.

**Specialty-aware hospital ranking** is a straightforward two-key sort, not ML:
`backend/app.py`'s `/hospitals` route sorts by
`(not matches_recommended_specialty, distance_km)` — Python's tuple comparison means
"matches" hospitals (False sorts before True) always come first as a group, sorted by
distance within each group. `disease_specialty.py` is a static, hand-curated
dict (disease → specialty), looked up case-insensitively.

**Why a desktop client instead of a web frontend:** stated rationale in the README/code
comments is that the ML/NLP stack (`scikit-learn`, `shap`, `sentence-transformers`) has
no mature JVM equivalent, so splitting it into its own service means the model can be
retrained/redeployed without touching the client — and a future web frontend could
call the same API unchanged. A secondary, implicit reason: JavaFX + Maven is a
recognizable "enterprise desktop" skill pairing that's genuinely different from yet
another React CRUD app, which stands out in a portfolio of five projects.

**Java–Python interop is plain HTTP/JSON**, nothing fancier (no gRPC, no shared schema
codegen) — `org.json` (`JSONObject`/`JSONArray`) on the Java side, Pydantic models
(`MatchRequest`, `PredictRequest`, `HospitalsRequest`) on the FastAPI side for request
validation. There's no shared contract/schema file between the two — a real
maintenance risk if either side drifts (see "What I'd improve").

---

## 4. What I verified end-to-end (2026-07-21)

- `ml/`: created a venv (Python 3.14), `pip install -r requirements.txt` succeeded
  clean. `python train.py` ran successfully — held-out test accuracy 97.62%, macro-F1
  98.37%, 5-fold CV macro-F1 1.0000 ± 0.0000 (matches the honest ceiling-effect caveat
  already in the README: the training data repeats a fixed symptom set per disease).
  `python build_symptom_embeddings.py` ran successfully (132 phrases → 384-dim
  embeddings). Manually exercised `SymptomMatcher.match()` with
  `"it burns when I pee and I am exhausted, also I have a high fever"` — correctly
  matched `burning_micturition` (0.756), `fatigue` (0.593), `high_fever` (0.868), zero
  unmatched segments.
- `backend/`: created a separate venv, installed requirements, started
  `uvicorn app:app` and hit all 5 routes with curl: `/health`, `/symptoms` (132 items),
  `/match-symptoms`, `/predict` (returned real SHAP contributions), `/hospitals` (with
  and without a `disease` filter — confirmed urology hospitals correctly outranked a
  closer oncology-only hospital for a UTI query, and confirmed a 400 error for an
  unknown symptom key).
- `desktop/`: `mvn -q compile` succeeded cleanly against `pom.xml` (JavaFX 26.0.1,
  `org.json` 20260522, Java 21 release target). Endpoint paths and JSON shapes in
  `ApiClient.java` match `app.py`'s routes exactly (`/health`, `/symptoms`,
  `/match-symptoms`, `/predict`, `/hospitals`) — no URL/port mismatch found.

**Bug found and fixed:** `backend/requirements.txt` did not list `scikit-learn`, even
though `app.py` unpickles a `RandomForestClassifier` via `joblib.load(...)` at
startup — that only worked by accident because `shap` transitively pulls in
`scikit-learn`. Added `scikit-learn` explicitly to `backend/requirements.txt` so the
backend's real dependency is declared rather than riding on another package's
transitive dependency (which could silently break if `shap` ever drops that
dependency or pins an incompatible version).

No other defects found: README setup steps match reality for all three tiers, CSV
column names/paths in `train.py` are correct, and there is no Java↔backend endpoint
or port mismatch.

---

## 5. 15+ interview questions with model answers

**ML / data**

1. **Q: Your 5-fold CV F1 is a perfect 1.0000 — isn't that a red flag?**
   A: Yes, and the README says so explicitly. `data/training_data.csv` repeats a fixed
   symptom combination per disease across many rows, so any reasonable classifier
   trivially separates the classes — it's a ceiling effect of a clean synthetic
   dataset, not evidence of real generalization. I report it alongside the held-out
   `test_data.csv` accuracy (97.6%) as the more honest (though still same-distribution)
   number, and I'm upfront that this is a systems/ML-pipeline project, not a validated
   diagnostic tool.

2. **Q: Is there train/test leakage?**
   A: `train.py` reads `training_data.csv` and `test_data.csv` as two physically
   separate files and never mixes them — CV is run only on the training split, and
   `test_data.csv` is evaluated once after fitting. So no leakage in the mechanical
   sense; the caveat is that both files are drawn from the same synthetic generation
   process, so "held-out" doesn't mean "out of distribution."

3. **Q: Why Random Forest and not a neural net or logistic regression?**
   A: 132 binary features → 41 classes, ~4900 rows, is a small tabular problem where
   tree ensembles typically match or beat neural nets and need far less tuning. It
   also pairs naturally with `shap.TreeExplainer`, which gives exact (not sampled/
   approximated) Shapley values for tree ensembles — that's explicitly why `train.py`'s
   comments call out `TreeExplainer` as "exact and fast... unlike KernelSHAP."

4. **Q: What does `class_weight="balanced"` do and why use it here?**
   A: It reweights each class inversely proportional to its frequency in `y_train`,
   so the loss doesn't implicitly favor whichever diseases have more rows. Given this
   dataset's rows-per-disease are fairly uniform it's a defensive default more than a
   load-bearing fix, but it's cheap insurance against imbalance.

5. **Q: How would you evaluate this more rigorously if it were a real product?**
   A: Get a genuinely independent dataset (different source, ideally real de-identified
   patient-reported symptoms, not the same generator), evaluate per-class
   precision/recall (macro-F1 can hide a badly-performing rare class), and add a
   calibration check (reliability diagram) since the confidence score is shown to the
   user directly as a percentage in `SymptomCheckerPane`'s `ProgressBar`.

6. **Q: Walk me through what SHAP is actually computing here.**
   A: For each disease class, tree SHAP decomposes that specific prediction's
   probability into additive per-feature contributions relative to a baseline (the
   expected value over the training set) — `explainer.shap_values(x_row)` in
   `app.py` returns a `(1, n_features, n_classes)` array; `/predict` slices out
   `sv[0, :, class_index]` for the predicted class and ranks only the *present*
   symptoms (`present_idx`) by contribution, so the user sees why their reported
   symptoms drove that specific disease, not a generic global ranking.

7. **Q: Global vs. per-prediction SHAP — where's each used?**
   A: `train.py` computes one *global* importance ranking once, over a 400-row sample,
   averaged over `|shap_values|` across samples and classes — saved as
   `top_20_global_symptoms_by_shap` in `metrics.json`, useful for a model card / feature
   audit. `app.py`'s `/predict` computes a fresh, *local* explanation per request for
   the specific symptoms and class in question — that's the one actually surfaced to
   end users in the desktop UI's "Why:" section.

**NLP / symptom matching**

8. **Q: Why cosine similarity over a fine-tuned classifier for symptom matching?**
   A: There's no labeled dataset of (free-text symptom description → canonical
   symptom key) to fine-tune on. Off-the-shelf sentence embeddings plus cosine
   similarity against a small (132-item) anchor set needs zero training data, and —
   important for a medical-adjacent tool — every match is inspectable: you get the
   exact anchor phrase and a numeric similarity score, versus a fine-tuned model's
   opaque decision.

9. **Q: Why threshold at 0.42 specifically, and what happens below it?**
   A: It's an empirically-chosen cutoff (see `MATCH_THRESHOLD` in `symptom_matcher.py`)
   below which a match is judged unreliable — the segment is put in
   `unmatched_segments` instead of being force-matched to the nearest (but poor) anchor,
   which the UI surfaces as "Couldn't confidently match: ..." rather than silently
   guessing wrong.

10. **Q: How does the matcher handle a sentence describing multiple symptoms?**
    A: `_segments()` splits on commas, "and"/"also"/"plus" (word-boundary regex,
    case-insensitive), semicolons and newlines, then each segment is embedded and
    matched independently — so "it burns when I pee and I'm exhausted" becomes two
    segments matched separately to `burning_micturition` and `fatigue`.

**System design**

11. **Q: How would you scale the API for concurrent users?**
    A: Right now the model, matcher, and hospital data are loaded once as module-level
    globals at import time (good — no per-request reload), and FastAPI/uvicorn already
    handles concurrent requests via async workers; I'd add multiple uvicorn workers
    (`--workers N`) behind a reverse proxy for real concurrency, since sklearn's
    `predict_proba` and SHAP's `shap_values` calls are synchronous/CPU-bound and would
    block an event loop worker otherwise.

12. **Q: How would you version the model in production?**
    A: `metrics.json`, `symptoms.json`, `diseases.json` are already separate artifacts
    from `model.joblib` — I'd add a version/hash field to `metrics.json`, keep old
    `model.joblib` files under versioned filenames or a model registry, and have
    `/health` report the loaded model's version so a client can detect a mismatch after
    a redeploy.

13. **Q: What happens if the backend is down when the desktop client starts?**
    A: `MedPredictApp.start()` calls `api.healthCheck()` and shows a non-blocking
    `Alert.WARNING` if it fails, then opens the app anyway — the user finds out on
    first actual request instead of being locked out, which is a reasonable
    degrade-gracefully choice for a desktop tool.

14. **Q: Why CORS `allow_origins=["*"]` — is that a security concern?**
    A: It's wide open, which is fine for a local single-user desktop client hitting
    `127.0.0.1`, but would need to be scoped to specific origins before this API
    was ever exposed beyond localhost — worth flagging unprompted in a review.

**Java**

15. **Q: Why Maven instead of Gradle for the desktop client?**
    A: Maven's declarative XML plus the `javafx-maven-plugin` (configured in `pom.xml`
    with `mainClass = com.medpredict.MedPredictApp`) gives a one-command
    `mvn javafx:run` that handles the JavaFX module path automatically — less
    boilerplate than wiring Gradle's JavaFX plugin by hand for a project this size.

16. **Q: Why JavaFX instead of Swing?**
    A: JavaFX has proper CSS-style styling, an FXML/scene-graph model, and better
    layout containers (`TableView`, `FlowPane`, `SplitPane`, all used here) — Swing is
    legacy and harder to make look non-1998.

17. **Q: How does the client avoid freezing the UI during network calls?**
    A: Every API call in `SymptomCheckerPane`/`HospitalLocatorPane` is wrapped in a raw
    `new Thread(() -> {...}).start()`, with the result marshaled back to the FX
    Application Thread via `Platform.runLater(...)` — necessary because JavaFX scene
    graph mutations must happen on the FX thread. The honest gap: it's a new `Thread`
    per call, not a bounded executor, which is fine here given the click-driven,
    low-frequency call pattern but wouldn't scale to rapid-fire requests.

18. **Q: Why force HTTP/1.1 in `ApiClient`?**
    A: Called out directly in a code comment: the JDK `HttpClient`'s default attempt to
    upgrade to HTTP/2 over plain `http://` could silently drop the POST body against
    a plain HTTP/1.1 server like uvicorn, manifesting as a confusing 422 "field
    required" server-side even though the client did send a body. Pinning
    `HttpClient.Version.HTTP_1_1` sidesteps that entirely.

---

## 6. What I'd improve (honest)

- **Testing**: there are no automated tests anywhere in the repo (no `pytest`/JUnit).
  I'd add unit tests for `symptom_matcher.py`'s segmentation/threshold logic,
  `disease_specialty.py`'s lookup, and the `/hospitals` sort order, plus an API
  integration test hitting a running FastAPI test client.
- **Model evaluation**: as noted above, both training and test data come from the
  same synthetic generator — I'd want a genuinely independent evaluation set before
  calling this more than a pipeline demo.
- **Containerization**: nothing is Dockerized. I'd containerize the backend (with the
  model artifacts baked in or mounted) and document a `docker-compose` for
  backend + a documented model-training step, so the whole thing runs with one command
  instead of three manual `pip install`/`mvn` steps across tiers.
- **`hospitals.json` is a static, hand-typed 15-hospital sample** for one city
  (Bengaluru) with hardcoded lat/lon — I'd replace it with a real datastore (even
  SQLite) and a geocoding lookup so it isn't limited to one hard-coded metro area.
- **No shared API schema/contract** between the Java and Python sides — request/response
  shapes are duplicated by hand (Pydantic models in `app.py`, manual `JSONObject`
  building in `ApiClient.java`). An OpenAPI spec (FastAPI generates one for free at
  `/docs`/`/openapi.json`) plus a generated Java client would remove that drift risk.
- **CORS is wide open** (`allow_origins=["*"]`) — fine for localhost-only use today,
  but would need real origin scoping before being exposed on a network.
