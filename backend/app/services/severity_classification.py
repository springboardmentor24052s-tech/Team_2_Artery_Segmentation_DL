
"""
Experimental stenosis severity classification.

Thresholds are provisional engineering settings.
They are NOT clinically validated diagnostic criteria.
"""

SEVERITY_THRESHOLDS = {
    "normal": (0.0, 25.0),
    "mild": (25.0, 50.0),
    "moderate": (50.0, 70.0),
    "severe": (70.0, 90.0),
    "critical": (90.0, 100.0),
}


def classify_stenosis_severity(narrowing_percent):
    """
    Classify an experimental narrowing percentage.

    Returns an experimental category and a validation warning.
    """

    if narrowing_percent is None:
        raise ValueError("Narrowing percentage cannot be None.")

    value = float(narrowing_percent)

    if not 0.0 <= value <= 100.0:
        raise ValueError(
            "Narrowing percentage must be between 0 and 100."
        )

    for severity, (lower, upper) in SEVERITY_THRESHOLDS.items():
        if lower <= value < upper:
            category = severity
            break
    else:
        category = "critical"

    return {
        "severity": category,
        "estimated_narrowing_percent": round(value, 2),
        "classification_type": "experimental",
        "thresholds": "provisional",
        "clinically_validated": False,
        "notice": (
            "Experimental severity category based on provisional "
            "percentage bands. Not a clinical diagnosis."
        )
    }
