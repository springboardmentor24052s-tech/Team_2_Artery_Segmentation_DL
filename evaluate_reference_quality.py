
import json
from pathlib import Path


IMAGE_ID = "74996d2933414f26a69c7dfa254ad28f"

CANDIDATE_PATH = Path(
    f"backend/results/{IMAGE_ID}_narrowing_analysis.json"
)

OUTPUT_PATH = Path(
    f"backend/results/{IMAGE_ID}_stenosis_percentage_analysis.json"
)


if not CANDIDATE_PATH.exists():
    raise FileNotFoundError(
        f"Narrowing report not found: {CANDIDATE_PATH}"
    )

with open(CANDIDATE_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)


results = []

print("\n========== REFERENCE DIAMETER QUALITY ==========")
print("Image ID:", IMAGE_ID)

for item in data["branches"]:

    branch_id = item["branch_id"]
    candidate = item["deepest_candidate"]

    left = float(candidate["left_reference"])
    right = float(candidate["right_reference"])
    minimum = float(candidate["smoothed_diameter"])

    reference = (left + right) / 2.0

    if reference <= 0:
        continue

    asymmetry = abs(left - right) / reference * 100

    narrowing = max(
        0.0,
        (1.0 - minimum / reference) * 100
    )

    if asymmetry <= 15:
        quality = "relatively_consistent"
    elif asymmetry <= 30:
        quality = "needs_review"
    else:
        quality = "poor"

    result = {
        "branch_id": branch_id,
        "candidate_position": candidate["position"],
        "x": candidate["x"],
        "y": candidate["y"],
        "minimum_diameter_pixels": round(minimum, 3),
        "left_reference_pixels": round(left, 3),
        "right_reference_pixels": round(right, 3),
        "reference_diameter_pixels": round(reference, 3),
        "reference_asymmetry_percent": round(asymmetry, 2),
        "estimated_narrowing_percent": round(narrowing, 2),
        "reference_quality": quality
    }

    results.append(result)

    print(f"\nBranch {branch_id}")
    print("Candidate diameter:", minimum)
    print("Reference diameter:", round(reference, 3))
    print("Reference asymmetry:", round(asymmetry, 2), "%")
    print("Experimental narrowing:", round(narrowing, 2), "%")
    print("Reference quality:", quality)


report = {
    "image_id": IMAGE_ID,
    "measurement_unit": "pixels",
    "geometry_mask_model": data.get("geometry_mask_model", "syntax"),
    "measurement_method": data.get(
        "measurement_method",
        "2 x Euclidean distance transform"
    ),
    "experimental": True,
    "clinically_validated": False,
    "branches_evaluated": len(results),
    "results": results
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n========== SUMMARY ==========")
print("Branches evaluated:", len(results))
print(
    "Consistent references:",
    sum(r["reference_quality"] == "relatively_consistent" for r in results)
)
print(
    "References needing review:",
    sum(r["reference_quality"] == "needs_review" for r in results)
)
print(
    "Poor references:",
    sum(r["reference_quality"] == "poor" for r in results)
)

print("\nSaved report:", OUTPUT_PATH)
print("\nPercentage narrowing analysis completed.")
