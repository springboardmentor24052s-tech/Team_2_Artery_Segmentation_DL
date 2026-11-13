
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_ID = "74996d2933414f26a69c7dfa254ad28f"

INPUT_PATH = Path(
    f"backend/results/{IMAGE_ID}_branch_diameter.json"
)

OUTPUT_PATH = Path(
    f"backend/results/{IMAGE_ID}_narrowing_analysis.json"
)


# ============================================================
# LOAD DIAMETER DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Branch diameter report not found: {INPUT_PATH}. "
        "Run test_branch_diameter.py first."
    )

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)

if data.get("image_id") != IMAGE_ID:
    raise ValueError("Image ID mismatch in diameter report.")


# ============================================================
# NARROWING CANDIDATE ANALYSIS
# ============================================================

def analyze_branch(branch, smoothing_size=5, reference_window=10):

    profile = branch["diameter_profile"]

    diameters = np.array(
        [p["diameter_pixels"] for p in profile],
        dtype=float
    )

    if len(diameters) < 2 * reference_window + 5:
        return None

    # Smooth small pixel-level fluctuations.
    smoothed = median_filter(
        diameters,
        size=smoothing_size,
        mode="nearest"
    )

    candidates = []

    # Exclude branch ends because they may be affected by
    # junctions or incomplete vessel segmentation.
    for i in range(
        reference_window,
        len(smoothed) - reference_window
    ):

        left_reference = np.median(
            smoothed[i-reference_window:i]
        )

        right_reference = np.median(
            smoothed[i+1:i+1+reference_window]
        )

        if left_reference <= 0 or right_reference <= 0:
            continue

        reference_diameter = (
            left_reference + right_reference
        ) / 2.0

        minimum_diameter = smoothed[i]

        if (
            minimum_diameter < left_reference
            and minimum_diameter < right_reference
        ):
            candidates.append({
                "position": i,
                "x": profile[i]["x"],
                "y": profile[i]["y"],
                "smoothed_diameter": round(
                    float(minimum_diameter), 3
                ),
                "left_reference": round(
                    float(left_reference), 3
                ),
                "right_reference": round(
                    float(right_reference), 3
                ),
                "combined_reference": round(
                    float(reference_diameter), 3
                )
            })

    if not candidates:
        return None

    # Keep the deepest candidate for initial inspection.
    best = min(
        candidates,
        key=lambda item: item["smoothed_diameter"]
    )

    return {
        "branch_id": branch["branch_id"],
        "branch_length": len(profile),
        "candidate_count": len(candidates),
        "deepest_candidate": best
    }


# ============================================================
# PROCESS ALL BRANCHES
# ============================================================

results = []

for branch in data["branches"]:

    analysis = analyze_branch(branch)

    if analysis is not None:
        results.append(analysis)


# ============================================================
# SAVE IMAGE-SPECIFIC RESULTS
# ============================================================

report = {
    "image_id": IMAGE_ID,
    "input_image_task": data.get("input_image_task", "stenosis"),
    "geometry_mask_model": data.get("geometry_mask_model", "syntax"),
    "measurement_unit": data.get("measurement_unit", "pixels"),
    "measurement_method": data.get(
        "measurement_method",
        "2 x Euclidean distance transform"
    ),
    "smoothing_window": 5,
    "reference_window": 10,
    "branches_analyzed": len(results),
    "branches": results,
    "experimental": True,
    "clinical_validation": False
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== NARROWING CANDIDATE ANALYSIS ==========")

print("Image ID:", IMAGE_ID)
print("Branches analyzed:", len(results))

for item in results:

    candidate = item["deepest_candidate"]

    print(f"\nBranch {item['branch_id']}")
    print("Branch length:", item["branch_length"])
    print("Local minima found:", item["candidate_count"])
    print("Candidate position:", candidate["position"])
    print(
        "Image coordinates:",
        candidate["x"],
        candidate["y"]
    )
    print(
        "Smoothed diameter:",
        candidate["smoothed_diameter"]
    )
    print("Left reference:", candidate["left_reference"])
    print("Right reference:", candidate["right_reference"])
    print(
        "Combined reference:",
        candidate["combined_reference"]
    )

print("\n========== OUTPUT FILE ==========")
print("Saved report:", OUTPUT_PATH)
print("\nNarrowing candidate analysis completed.")
print("Note: Results are experimental pixel-based estimates.")
