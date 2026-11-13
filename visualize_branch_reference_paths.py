
import json
import cv2
import numpy as np
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
    "backend/results/branch_reference_path_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

with open(DIAMETER_PATH, "r", encoding="utf-8") as f:
    diameter_data = json.load(f)

with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
    candidate_data = json.load(f)

branches = {
    branch["branch_id"]: branch
    for branch in diameter_data["branches"]
}

for item in candidate_data:

    branch_id = item["branch_id"]
    candidate = item["deepest_candidate"]

    profile = branches[branch_id]["diameter_profile"]
    position = candidate["position"]

    output = image.copy()

    # Full branch path.
    all_points = np.array([
        [p["x"], p["y"]] for p in profile
    ], dtype=np.int32)

    if len(all_points) >= 2:
        cv2.polylines(
            output,
            [all_points.reshape(-1, 1, 2)],
            False,
            (255, 255, 0),
            1
        )

    # Reference windows: 10 positions on either side.
    left_points = np.array([
        [p["x"], p["y"]]
        for p in profile[max(0, position - 10):position]
    ], dtype=np.int32)

    right_points = np.array([
        [p["x"], p["y"]]
        for p in profile[position + 1:position + 11]
    ], dtype=np.int32)

    if len(left_points) >= 2:
        cv2.polylines(
            output,
            [left_points.reshape(-1, 1, 2)],
            False,
            (0, 255, 0),
            3
        )

    if len(right_points) >= 2:
        cv2.polylines(
            output,
            [right_points.reshape(-1, 1, 2)],
            False,
            (255, 0, 0),
            3
        )

    # Candidate location.
    x = candidate["x"]
    y = candidate["y"]

    cv2.circle(
        output,
        (x, y),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        output,
        f"Branch {branch_id}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    output_path = (
        OUTPUT_DIR / f"branch_{branch_id}_path.png"
    )

    cv2.imwrite(str(output_path), output)

print("Visualizations saved:", OUTPUT_DIR)
print("Branches visualized:", len(candidate_data))
