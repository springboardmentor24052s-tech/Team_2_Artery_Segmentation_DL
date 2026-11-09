
import cv2
import numpy as np


def postprocess_mask(mask, min_component_area=20, kernel_size=3):
    if mask is None or mask.size == 0:
        raise ValueError("Input mask is empty.")

    if min_component_area < 1:
        raise ValueError("Minimum component area must be positive.")

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    original_pixels = int(np.count_nonzero(binary))

    # Close small gaps in the segmented vessel regions.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    closed_mask = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Identify connected components.
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            closed_mask,
            connectivity=8
        )
    )

    cleaned_mask = np.zeros_like(binary)
    components = []

    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])

        if area < min_component_area:
            continue

        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])

        cleaned_mask[labels == index] = 255

        components.append({
            "component_id": len(components) + 1,
            "area_pixels": area,
            "bounding_box": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "centroid": {
                "x": round(float(centroids[index][0]), 2),
                "y": round(float(centroids[index][1]), 2)
            }
        })

    cleaned_pixels = int(np.count_nonzero(cleaned_mask))

    return {
        "mask": cleaned_mask,
        "original_foreground_pixels": original_pixels,
        "cleaned_foreground_pixels": cleaned_pixels,
        "removed_pixels": max(0, original_pixels - cleaned_pixels),
        "component_count": len(components),
        "components": components
    }
