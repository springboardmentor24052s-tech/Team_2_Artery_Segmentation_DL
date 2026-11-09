
from pathlib import Path

import cv2
import numpy as np


def localize_stenosis_candidates(
    original_image_path,
    candidates,
    output_path
):
    """
    Draw bounding boxes around extracted stenosis candidates
    on the original angiography image.
    """

    original_image_path = Path(original_image_path)
    output_path = Path(output_path)

    # Validate original image
    if not original_image_path.is_file():
        raise FileNotFoundError(
            f"Original image not found: {original_image_path}"
        )

    image = cv2.imread(
        str(original_image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Unable to read the original angiography image."
        )

    if not isinstance(candidates, list):
        raise ValueError("Candidates must be provided as a list.")

    image_height, image_width = image.shape[:2]

    localized_candidates = []

    # Draw bounding boxes
    for candidate in candidates:

        candidate_id = candidate.get("candidate_id")

        bounding_box = candidate.get("bounding_box")

        if not isinstance(bounding_box, dict):
            raise ValueError(
                f"Invalid bounding box for candidate {candidate_id}."
            )

        try:
            x = int(bounding_box["x"])
            y = int(bounding_box["y"])
            width = int(bounding_box["width"])
            height = int(bounding_box["height"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Invalid bounding box coordinates for candidate {candidate_id}."
            )

        if width <= 0 or height <= 0:
            continue

        # Clip coordinates to image boundaries
        x1 = max(0, min(x, image_width - 1))
        y1 = max(0, min(y, image_height - 1))
        x2 = max(0, min(x + width, image_width - 1))
        y2 = max(0, min(y + height, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        mean_probability = float(
            candidate.get("mean_probability", 0.0)
        )

        # Draw bounding rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Add candidate label
        label = (
            f"Candidate {candidate_id} "
            f"({mean_probability:.2f})"
        )

        label_y = max(20, y1 - 8)

        cv2.putText(
            image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        localized_candidates.append({
            "candidate_id": candidate_id,
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "mean_probability": round(mean_probability, 4)
        })

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save localized image
    saved = cv2.imwrite(
        str(output_path),
        image
    )

    if not saved or not output_path.is_file():
        raise RuntimeError(
            "Unable to save localized angiography image."
        )

    return {
        "status": "success",
        "image_width": image_width,
        "image_height": image_height,
        "localized_candidate_count": len(localized_candidates),
        "localized_candidates": localized_candidates,
        "output_path": str(output_path.resolve())
    }
