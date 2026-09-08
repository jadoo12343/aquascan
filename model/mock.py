"""
AquaScan — Mock Classification Pipeline (Person B stub)
======================================================
This module provides a drop-in replacement for Person A's real
EfficientNetB0 model so Person B can build, test, and demo the entire
Streamlit reporting flow before the trained weights arrive (Day 7).

How to swap in Person A's real model on Day 7
---------------------------------------------
1. Person A creates model/predict.py with a predict_image() function
   that matches the same signature as mock_predict() below.
2. Change the import in app/app.py from:
       from model.mock import mock_predict as predict_image
   to:
       from model.predict import predict_image
3. Done — zero other changes required.
"""

import random
from PIL import Image

# ── Target classes AquaScan recognises ──────────────────────────────────────
WASTE_CLASSES = ["plastic", "metal", "glass", "cardboard", "paper", "trash"]

# ── Severity lookup table (agreed with Person A as the "handoff contract") ──
SEVERITY_MAP = {
    "plastic":   "Critical",
    "metal":     "High",
    "glass":     "High",
    "cardboard": "Medium",
    "paper":     "Low",
    "trash":     "Medium",
}


def mock_predict(image: Image.Image) -> tuple[str, float, str]:
    """
    Simulate an EfficientNetB0 classification result.

    Args:
        image: PIL Image uploaded by the user (not used by the mock,
               but kept in the signature so the swap on Day 7 is seamless).

    Returns:
        predicted_class (str): One of WASTE_CLASSES
        confidence      (float): Realistic probability 0.72 – 0.98
        severity        (str): 'Critical' | 'High' | 'Medium' | 'Low'
    """
    predicted_class = random.choice(WASTE_CLASSES)
    confidence = round(random.uniform(0.72, 0.98), 2)
    severity = SEVERITY_MAP[predicted_class]
    return predicted_class, confidence, severity
