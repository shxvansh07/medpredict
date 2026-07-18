"""
Maps each predicted disease to the hospital specialty department most relevant
to it, so the hospital locator can rank hospitals by relevance-then-distance
instead of distance alone. Lookup is case-insensitive on the disease name.
"""

_MAP = {
    "fungal infection": "dermatology",
    "allergy": "general_medicine",
    "gerd": "gastroenterology",
    "chronic cholestasis": "gastroenterology",
    "drug reaction": "dermatology",
    "peptic ulcer diseae": "gastroenterology",
    "aids": "infectious_disease",
    "diabetes": "endocrinology",
    "gastroenteritis": "gastroenterology",
    "bronchial asthma": "pulmonology",
    "hypertension": "cardiology",
    "migraine": "neurology",
    "cervical spondylosis": "orthopedics",
    "paralysis (brain hemorrhage)": "neurology",
    "jaundice": "gastroenterology",
    "malaria": "infectious_disease",
    "chicken pox": "infectious_disease",
    "dengue": "infectious_disease",
    "typhoid": "infectious_disease",
    "hepatitis a": "gastroenterology",
    "hepatitis b": "gastroenterology",
    "hepatitis c": "gastroenterology",
    "hepatitis d": "gastroenterology",
    "hepatitis e": "gastroenterology",
    "alcoholic hepatitis": "gastroenterology",
    "tuberculosis": "pulmonology",
    "common cold": "general_medicine",
    "pneumonia": "pulmonology",
    "dimorphic hemmorhoids(piles)": "general_medicine",
    "heart attack": "cardiology",
    "varicose veins": "general_medicine",
    "hypothyroidism": "endocrinology",
    "hyperthyroidism": "endocrinology",
    "hypoglycemia": "endocrinology",
    "osteoarthristis": "orthopedics",
    "arthritis": "orthopedics",
    "(vertigo) paroymsal  positional vertigo": "ENT",
    "acne": "dermatology",
    "urinary tract infection": "urology",
    "psoriasis": "dermatology",
    "impetigo": "dermatology",
}

FALLBACK_SPECIALTY = "general_medicine"


def specialty_for(disease: str) -> str:
    return _MAP.get(disease.strip().lower(), FALLBACK_SPECIALTY)
