"""
Precomputes sentence embeddings for every canonical symptom's natural-language
phrase, so the backend can do free-text -> symptom matching at request time
without re-embedding the whole symptom list on every call.

Run once after train.py (or whenever symptom_phrases.py changes):
    python build_symptom_embeddings.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from symptom_phrases import to_phrase

MODELS_DIR = Path(__file__).parent / "models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # same free local embedding model used in the RAG project


def main():
    symptoms = json.loads((MODELS_DIR / "symptoms.json").read_text())
    phrases = [to_phrase(s) for s in symptoms]

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(phrases, normalize_embeddings=True, show_progress_bar=True)

    np.save(MODELS_DIR / "symptom_embeddings.npy", embeddings.astype(np.float32))
    (MODELS_DIR / "symptom_phrases.json").write_text(json.dumps(phrases, indent=2))
    print(f"Embedded {len(phrases)} symptom phrases -> {embeddings.shape}")


if __name__ == "__main__":
    main()
