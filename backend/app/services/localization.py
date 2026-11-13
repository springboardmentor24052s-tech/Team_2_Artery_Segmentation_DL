
from pathlib import Path

import cv2


# Candidate colours in BGR format
CANDIDATE_COLORS = [
    (0, 0, 255),       # Red
    (255, 100, 0),     # Blue
    (0, 180, 0),       # Green
    (180, 0, 180),     # Purple
    (0, 165, 255),     # Orange
    (255, 200, 0),     # Cyan
    (180, 80, 180),    # Pink
    (0, 200, 200),     # Yellow
]


def rectangles_overlap(rect1, rect2, padding=3):
    """Check whether two rectangles overlap."""

    x1, y1, x2, y2 = rect1
    a1, b1, a2, b2 = rect2

    return not (
        x2 + padding <= a1
        or a2 + padding <= x1
        or y2 + padding <= b1
        or b2 + padding <= y1
    )


def calculate_label_position(
    x1,
    y1,
    x2,
    y2,
    label_width,
    label_height,
    image_width,
    image_height,
    occupied_labels,
    all_candidate_boxes,
):
    """
    Select a label position while avoiding other labels
    and other candidate bounding boxes where possible.
    """

    margin = 3
    gap = 5

    # Candidate positions around the current bounding box.
    positions = [
        (x1, y1 - label_height - gap),                  # Above
        (x1, y2 + gap),                                 # Below
        (x2 + gap, y1),                                 # Right
        (x1 - label_width - gap, y1),                   # Left
        (x2 + gap, y1 - label_height),                  # Upper right
        (x1 - label_width - gap, y2),                   # Lower left
        (x2 - label_width, y2 + gap),                   # Lower right
        (x1, y1 - label_height - gap - 8),              # Further above
    ]

    scored_positions = []

    for raw_x, raw_y in positions:

        # Keep the label inside the image.
        x = max(
            margin,
            min(raw_x, image_width - label_width - margin)
        )

        top = max(
            margin,
            min(raw_y, image_height - label_height - margin)
        )

        rectangle = (
            x,
            top,
            x + label_width,
            top + label_height
        )

        # Count collisions with already placed labels.
        label_collisions = sum(
            rectangles_overlap(rectangle, existing)
            for existing in occupied_labels
        )

        # Count collisions with other candidate boxes.
        box_collisions = sum(
            rectangles_overlap(rectangle, box, padding=2)
            for box in all_candidate_boxes
        )

        # Prefer positions with fewer collisions.
        # Distance is a tie-breaker to keep labels near their boxes.
        distance = abs(x - x1) + abs(top - y1)

        score = (
            label_collisions * 10000
            + box_collisions * 1000
            + distance
        )

        scored_positions.append(
            (score, x, top, rectangle)
        )

    scored_positions.sort(key=lambda item: item[0])

    _, best_x, best_top, best_rectangle = scored_positions[0]

    return best_x, best_top, best_rectangle


def localize_stenosis_candidates(
    original_image_path,
    candidates,
    output_path
):
    """
    Draw candidate bounding boxes and labels on the original
    angiography image.

    Label placement attempts to avoid overlapping labels
    and other candidate bounding boxes.

    Outputs are experimental candidate localizations,
    not confirmed stenosis diagnoses.
    """

    original_image_path = Path(original_image_path)
    output_path = Path(output_path)

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
        raise ValueError(
            "Candidates must be provided as a list."
        )

    image_height, image_width = image.shape[:2]

    # Validate and prepare candidates.
    prepared_candidates = []

    for index, candidate in enumerate(candidates):

        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be a dictionary.")

        candidate_id = candidate.get("candidate_id", index + 1)
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

        x1 = max(0, min(x, image_width - 1))
        y1 = max(0, min(y, image_height - 1))
        x2 = max(0, min(x + width, image_width - 1))
        y2 = max(0, min(y + height, image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        prepared_candidates.append({
            "original_index": index,
            "candidate": candidate,
            "candidate_id": candidate_id,
            "box": (x1, y1, x2, y2),
        })

    # Draw larger candidates first for label placement priority.
    drawing_order = sorted(
        prepared_candidates,
        key=lambda item: (
            int(item["candidate"].get("area_pixels", 0))
        ),
        reverse=True
    )

    all_candidate_boxes = [
        item["box"] for item in prepared_candidates
    ]

    occupied_labels = []
    localized_by_index = {}

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    for item in drawing_order:

        candidate = item["candidate"]
        candidate_id = item["candidate_id"]

        x1, y1, x2, y2 = item["box"]

        original_index = item["original_index"]

        color = CANDIDATE_COLORS[
            original_index % len(CANDIDATE_COLORS)
        ]

        mean_probability = float(
            candidate.get("mean_probability", 0.0)
        )

        maximum_probability = float(
            candidate.get(
                "maximum_probability",
                candidate.get(
                    "max_probability",
                    mean_probability
                )
            )
        )

        area_pixels = int(
            candidate.get("area_pixels", 0)
        )

        # Draw candidate bounding box.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f"Candidate {candidate_id} "
            f"({mean_probability:.2f})"
        )

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        label_height = text_height + baseline + 4
        label_width = text_width + 6

        label_x, label_top, label_rectangle = (
            calculate_label_position(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label_width=label_width,
                label_height=label_height,
                image_width=image_width,
                image_height=image_height,
                occupied_labels=occupied_labels,
                all_candidate_boxes=[
                    box for box in all_candidate_boxes
                    if box != item["box"]
                ],
            )
        )

        occupied_labels.append(label_rectangle)

        # Draw label background.
        cv2.rectangle(
            image,
            (
                label_rectangle[0],
                label_rectangle[1]
            ),
            (
                label_rectangle[2],
                label_rectangle[3]
            ),
            color,
            -1
        )

        # Draw white label text.
        cv2.putText(
            image,
            label,
            (
                label_x + 3,
                label_top + text_height + 1
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        localized_by_index[original_index] = {
            "candidate_id": candidate_id,
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "centroid": candidate.get("centroid"),
            "area_pixels": area_pixels,
            "mean_probability": round(mean_probability, 4),
            "maximum_probability": round(maximum_probability, 4),
            "label_position": {
                "x": int(label_x),
                "y": int(label_top)
            }
        }

    # Preserve the original candidate order in the API response.
    localized_candidates = [
        localized_by_index[index]
        for index in sorted(localized_by_index)
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
