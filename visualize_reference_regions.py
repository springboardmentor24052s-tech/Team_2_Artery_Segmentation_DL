
import json
import cv2
from pathlib import Path

IMAGE_PATH = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/test/images/1.png"
)

DIAMETER_PATH = Path(
    "backend/results/branch_diameter_test.json"
)

CANDIDATE_PATH = Path(
    "backend/results/narrowing_candidate_analysis.json"
)

OUTPUT_DIR = Path(
    "backend/results/reference_region_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

with open(DIAMETER_PATH, "r", encoding="utf-8") as file:
    diameter_data = json.load(file)

with open(CANDIDATE_PATH, "r", encoding="utf-8") as file:
    candidate_data = json.load(file)

diameter_by_id = {
    branch["branch_id"]: branch
    for branch in diameter_data["branches"]
}

for item in candidate_data:

    branch_id = item["branch_id"]
    candidate = item["deepest_candidate"]

    branch = diameter_by_id[branch_id]
    profile = branch["diameter_profile"]

    position = candidate["position"]

    left_start = max(0, position - 10)
    left_end = position

    right_start = position + 1
    right_end = min(len(profile), position + 11)

    output = image.copy()

    # Draw left reference region in green.
    for point in profile[left_start:left_end]:
        x, y = point["x"], point["y"]
        cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

    # Draw right reference region in blue.
    for point in profile[right_start:right_end]:
        x, y = point["x"], point["y"]
        cv2.circle(output, (x, y), 3, (255, 0, 0), -1)

    # Draw candidate in red.
    x, y = candidate["x"], candidate["y"]
    cv2.circle(output, (x, y), 6, (0, 0, 255), -1)

    cv2.putText(
        output,
        f"Branch {branch_id}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    output_path = OUTPUT_DIR / f"branch_{branch_id}_references.png"

    cv2.imwrite(str(output_path), output)

print("Reference visualizations saved:", OUTPUT_DIR)
print("Total branches visualized:", len(candidate_data))
