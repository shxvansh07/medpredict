"""
Human-readable phrase for every canonical symptom feature in the dataset.

The dataset's column names are underscore-joined tokens ("burning_micturition",
"toxic_look_(typhos)") -- fine for a checkbox UI, but useless as anchors for
semantic matching against a user's free-text description ("it burns when I pee").
This module supplies a natural-language phrase per symptom so the embedding
matcher (symptom_matcher.py) has something meaningful to compare against.

Only the non-obvious / jargon-heavy symptoms are overridden by hand; everything
else falls back to a cleaned version of the column name.
"""
import re

OVERRIDES = {
    "burning_micturition": "burning sensation while urinating",
    "spotting_ urination": "spotting or light bleeding during urination",
    "foul_smell_of urine": "foul smelling urine",
    "continuous_feel_of_urine": "constant feeling of needing to urinate",
    "toxic_look_(typhos)": "a very sick, drained, toxic appearance",
    "dischromic _patches": "discolored patches on the skin",
    "spinning_movements": "a spinning sensation, like the room is spinning",
    "loss_of_balance": "trouble keeping balance",
    "unsteadiness": "feeling unsteady on your feet",
    "weakness_of_one_body_side": "weakness on just one side of the body",
    "altered_sensorium": "confusion or an altered state of consciousness",
    "belly_pain": "pain in the belly or abdomen",
    "abnormal_menstruation": "irregular or abnormal periods",
    "polyuria": "urinating much more often than usual",
    "mucoid_sputum": "coughing up mucus-like phlegm",
    "rusty_sputum": "coughing up rust colored phlegm",
    "lack_of_concentration": "trouble concentrating or focusing",
    "visual_disturbances": "disturbances or changes in vision",
    "receiving_blood_transfusion": "recently received a blood transfusion",
    "receiving_unsterile_injections": "recently received an unsterile injection",
    "distention_of_abdomen": "a bloated, distended abdomen",
    "history_of_alcohol_consumption": "a history of regular alcohol consumption",
    "fluid_overload": "swelling from fluid buildup in the body",
    "fluid_overload.1": "swelling from fluid buildup in the body",
    "blood_in_sputum": "coughing up blood",
    "prominent_veins_on_calf": "visibly prominent, bulging veins on the calf",
    "painful_walking": "pain while walking",
    "pus_filled_pimples": "pimples filled with pus",
    "scurring": "scarring on the skin",
    "silver_like_dusting": "a silvery, flaky dusting on the skin",
    "small_dents_in_nails": "small pits or dents in the nails",
    "inflammatory_nails": "red, inflamed skin around the nails",
    "red_sore_around_nose": "a red sore around the nose",
    "yellow_crust_ooze": "a yellow, crusty ooze on the skin",
    "cold_hands_and_feets": "cold hands and feet",
    "swelled_lymph_nodes": "swollen lymph nodes",
    "swollen_extremeties": "swollen arms or legs",
    "extra_marital_contacts": "multiple or extramarital sexual contacts",
    "drying_and_tingling_lips": "dry, tingling lips",
    "internal_itching": "an internal itching sensation",
    "red_spots_over_body": "red spots spread over the body",
    "family_history": "a family history of this condition",
    "nodal_skin_eruptions": "small nodular eruptions on the skin",
    "patches_in_throat": "visible patches in the throat",
    "irregular_sugar_level": "unstable or irregular blood sugar levels",
    "watering_from_eyes": "watery, teary eyes",
    "acute_liver_failure": "sudden, severe liver failure",
    "swelling_of_stomach": "visible swelling of the stomach",
    "blurred_and_distorted_vision": "blurred or distorted vision",
    "weakness_in_limbs": "weakness in the arms or legs",
    "fast_heart_rate": "a noticeably fast heart rate",
    "pain_during_bowel_movements": "pain during bowel movements",
    "pain_in_anal_region": "pain in the anal region",
    "bloody_stool": "blood in the stool",
    "irritation_in_anus": "itching or irritation around the anus",
}


def to_phrase(symptom_key: str) -> str:
    if symptom_key in OVERRIDES:
        return OVERRIDES[symptom_key]
    cleaned = symptom_key.replace(".1", "").replace("_", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
