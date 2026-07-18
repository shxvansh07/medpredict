"""
Free-text -> canonical-symptom matcher.

The classic version of this project makes the user hunt through a checklist of
132 jargon-heavy symptom names. Here the user can instead type a plain-English
description ("burning when I pee, and I've been really tired and dizzy") and
the matcher segments it and semantically links each segment to the closest
canonical symptom via cosine similarity over sentence embeddings -- the same
embedding-matching technique used for retrieval in the RAG project, applied
here to entity linking instead of document search.

This is intentionally a lightweight, explainable approach (cosine similarity
over a fixed anchor set) rather than a black-box NER model: every match ships
with its similarity score and the exact phrase it was compared against, so the
UI can show *why* a segment was linked to a given symptom.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).parent / "models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

MATCH_THRESHOLD = 0.42  # below this, we report the segment as unmatched rather than guess

_SPLIT_RE = re.compile(r",|\band\b|\balso\b|\bplus\b|\n|;", flags=re.IGNORECASE)


@dataclass
class SymptomMatch:
    segment: str
    symptom: str
    phrase: str
    similarity: float


@dataclass
class MatchResult:
    matched: list       # list[SymptomMatch]
    unmatched_segments: list  # list[str]


class SymptomMatcher:
    def __init__(self):
        self.symptoms = json.loads((MODELS_DIR / "symptoms.json").read_text())
        self.phrases = json.loads((MODELS_DIR / "symptom_phrases.json").read_text())
        self.embeddings = np.load(MODELS_DIR / "symptom_embeddings.npy")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def _segments(self, text: str) -> list:
        parts = _SPLIT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def match(self, free_text: str) -> MatchResult:
        segments = self._segments(free_text)
        if not segments:
            return MatchResult(matched=[], unmatched_segments=[])

        seg_embeddings = self.model.encode(segments, normalize_embeddings=True)
        sims = seg_embeddings @ self.embeddings.T  # (n_segments, n_symptoms) cosine sim (both normalized)

        matched, unmatched = [], []
        seen_symptoms = set()
        for i, segment in enumerate(segments):
            best_j = int(np.argmax(sims[i]))
            score = float(sims[i][best_j])
            if score >= MATCH_THRESHOLD:
                symptom = self.symptoms[best_j]
                if symptom not in seen_symptoms:
                    matched.append(SymptomMatch(
                        segment=segment,
                        symptom=symptom,
                        phrase=self.phrases[best_j],
                        similarity=round(score, 3),
                    ))
                    seen_symptoms.add(symptom)
            else:
                unmatched.append(segment)

        return MatchResult(matched=matched, unmatched_segments=unmatched)
