"""
Trains the MedPredict symptom-to-condition classifier.

Model: RandomForestClassifier over 132 binary symptom features -> 41 conditions.
Produces, under models/:
  - model.joblib        the trained classifier
  - symptoms.json       ordered list of the 132 canonical symptom feature names
  - diseases.json       ordered list of the class labels (as the model sees them)
  - metrics.json        held-out test accuracy/F1 + global SHAP feature ranking

Run: python train.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LABEL_COL = "prognosis"
DROP_COLS = [LABEL_COL, "Unnamed: 133"]


def load_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    y = df[LABEL_COL].str.strip()
    X = df.drop(columns=[LABEL_COL])
    return X, y


def main():
    t0 = time.time()

    X_train, y_train = load_split(DATA_DIR / "training_data.csv")
    X_test, y_test = load_split(DATA_DIR / "test_data.csv")

    symptoms = list(X_train.columns)
    assert list(X_test.columns) == symptoms, "train/test symptom columns must match"

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold CV on the training split, as a robustness check independent of the
    # (small, one-row-per-disease) held-out test file.
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")

    clf.fit(X_train, y_train)

    test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average="macro")

    # --- Explainability: global SHAP feature importance -------------------------
    # TreeExplainer is exact and fast for tree ensembles (no sampling/approximation
    # needed, unlike KernelSHAP) -- this is what the backend's /predict endpoint
    # uses per-request to explain *individual* predictions, not just this global
    # summary computed once at training time.
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train.iloc[:400])  # sample for speed
    # shap_values: (n_samples, n_features, n_classes) in recent SHAP versions
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    global_importance = sorted(
        zip(symptoms, mean_abs_shap.tolist()), key=lambda p: p[1], reverse=True
    )

    diseases = sorted(clf.classes_.tolist())

    joblib.dump(clf, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "symptoms.json").write_text(json.dumps(symptoms, indent=2))
    (MODELS_DIR / "diseases.json").write_text(json.dumps(diseases, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps({
        "held_out_test_accuracy": test_accuracy,
        "held_out_test_f1_macro": test_f1_macro,
        "cv_f1_macro_mean": float(cv_scores.mean()),
        "cv_f1_macro_std": float(cv_scores.std()),
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "n_symptoms": len(symptoms),
        "n_diseases": len(diseases),
        "top_20_global_symptoms_by_shap": global_importance[:20],
        "trained_in_seconds": round(time.time() - t0, 1),
    }, indent=2))

    print(f"Held-out test accuracy: {test_accuracy:.4f}")
    print(f"Held-out test F1 (macro): {test_f1_macro:.4f}")
    print(f"5-fold CV F1 (macro): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Saved model + metadata to {MODELS_DIR}")


if __name__ == "__main__":
    main()
