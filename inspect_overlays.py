
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("backend/app/data/arcade/arcade/arcade")
MASK_ROOT = Path("backend/results/arcade_masks")
OUTPUT = Path("backend/results/mask_preview")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = [
    ("syntax", "train", "922.png"),
    ("stenosis", "train", "676.png"),
]


def add_title(image, title):
    result = image.copy()
    cv2.putText(
        result, title, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    return result


for task, split, filename in samples:
    image_path = ROOT / task / split / "images" / filename
    mask_path = (
        MASK_ROOT / task / split /
        f"{Path(filename).stem}_mask.png"
    )

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Missing image or mask:", task, filename)
        continue

    if image.shape[:2] != mask.shape:
        print("DIMENSION MISMATCH:", task, filename)
        continue

    # Show mask in grayscale.
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Green means annotated foreground.
    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    panels = [
        add_title(image, "Original"),
        add_title(mask_bgr, "Binary Mask"),
        add_title(overlay, "Green Overlay"),
    ]

    comparison = np.hstack(panels)

    output_path = OUTPUT / f"{task}_{Path(filename).stem}_comparison.png"
    cv2.imwrite(str(output_path), comparison)

    print(f"\nTask: {task}")
    print("Image:", image_path)
    print("Mask:", mask_path)
    print("Foreground pixels:", int(np.count_nonzero(mask)))
    print("Saved comparison:", output_path.resolve())

    cv2.imshow(f"{task.upper()} - Original | Mask | Overlay", comparison)
    print("Press any key in the image window to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nVisual inspection complete.")
