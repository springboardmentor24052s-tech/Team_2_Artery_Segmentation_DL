
import cv2
import numpy as np

from app.services.branch_geometry import extract_vessel_branches


def calculate_branch_diameters(cleaned_mask, centerline_mask):
    """
    Calculate diameter profiles for extracted vessel branches.

    Diameter is estimated as twice the distance from a skeleton
    pixel to the nearest background pixel.

    Coordinates are represented as (y, x).
    """

    if cleaned_mask is None or cleaned_mask.size == 0:
        raise ValueError("Cleaned vessel mask is empty.")

    if centerline_mask is None or centerline_mask.size == 0:
        raise ValueError("Centerline mask is empty.")

    if len(cleaned_mask.shape) == 3:
        cleaned_mask = cv2.cvtColor(
            cleaned_mask, cv2.COLOR_BGR2GRAY
        )

    if len(centerline_mask.shape) == 3:
        centerline_mask = cv2.cvtColor(
            centerline_mask, cv2.COLOR_BGR2GRAY
        )

    if cleaned_mask.shape != centerline_mask.shape:
        raise ValueError("Mask and centerline dimensions do not match.")

    binary = (cleaned_mask > 0).astype(np.uint8)
    skeleton = centerline_mask > 0

    distance_map = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    branch_result = extract_vessel_branches(centerline_mask)

    branch_profiles = []

    for branch in branch_result["branches"]:

        profile = []

        for y, x in branch["coordinates"]:

            radius = float(distance_map[y, x])
            diameter = radius * 2.0

            profile.append({
                "x": int(x),
                "y": int(y),
                "radius_pixels": round(radius, 3),
                "diameter_pixels": round(diameter, 3)
            })

        diameter_values = [
            point["diameter_pixels"]
            for point in profile
        ]

        if diameter_values:
            mean_diameter = float(np.mean(diameter_values))
            min_diameter = float(np.min(diameter_values))
            max_diameter = float(np.max(diameter_values))
        else:
            mean_diameter = 0.0
            min_diameter = 0.0
            max_diameter = 0.0

        branch_profiles.append({
            "branch_id": branch["branch_id"],
            "start_node": branch["start_node"],
            "end_node": branch["end_node"],
            "pixel_count": len(profile),
            "mean_diameter_pixels": round(mean_diameter, 3),
            "minimum_diameter_pixels": round(min_diameter, 3),
            "maximum_diameter_pixels": round(max_diameter, 3),
            "diameter_profile": profile
        })

    return {
        "branch_count": len(branch_profiles),
        "diameter_unit": "pixels",
        "measurement_method": "2 x Euclidean distance transform",
        "branches": branch_profiles
    }
