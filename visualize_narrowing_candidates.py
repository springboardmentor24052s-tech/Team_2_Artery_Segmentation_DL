
import cv2
import json
from pathlib import Path


IMAGE_PATH = Path(
    "backend/app/data/arcade/arcade/arcade/"
    "stenosis/test/images/1.png"
)

CANDIDATE_PATH = Path(
    "backend/results/narrowing_candidate_analysis.json"
)

OUTPUT_PATH = Path(
    "backend/results/narrowing_candidates_visualization.png"
)

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

with open(CANDIDATE_PATH, "r", encoding="utf-8") as file:
    candidates = json.load(file)

for item in candidates:

    branch_id = item["branch_id"]
    candidate = item["deepest_candidate"]

    x = int(candidate["x"])
    y = int(candidate["y"])

    # Mark candidate location.
    cv2.circle(
        image,
        (x, y),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        image,
        f"B{branch_id}",
        (x + 7, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

cv2.imwrite(str(OUTPUT_PATH), image)

print("Candidates visualized:", len(candidates))
print("Saved:", OUTPUT_PATH)
