"""
AquaScan — Severity Lookup
Rule-based severity mapping from predicted class to a priority label.
Kept separate from the classifier since this is a business decision,
not something the model itself predicts.
"""

SEVERITY_MAP = {
    "plastic":   "Critical",
    "metal":     "High",
    "glass":     "High",
    "cardboard": "Medium",
    "paper":     "Low",
    "trash":     "Medium",
}


def get_severity(predicted_class: str) -> str:
    """Look up a severity label. Defaults to 'Medium' if unrecognized."""
    return SEVERITY_MAP.get(predicted_class.lower(), "Medium")
