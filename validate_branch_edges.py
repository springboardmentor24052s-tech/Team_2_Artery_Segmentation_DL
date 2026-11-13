
import sys
import json
from pathlib import Path
from collections import Counter

import cv2

sys.path.insert(0, "backend")

from app.services.centerline import extract_centerline
from app.services.branch_geometry import (
    build_skeleton_graph,
    extract_vessel_branches
)

IMAGE_ID = "16c135650bb24194a3abfd836c03237c"

mask_path = (
    Path("backend/results/api_predictions")
    / f"{IMAGE_ID}_cleaned_mask.png"
)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(mask_path)

centerline = extract_centerline(mask)["centerline_mask"]

graph = build_skeleton_graph(centerline)
result = extract_vessel_branches(centerline)

def edge_key(a, b):
    return tuple(sorted((tuple(a), tuple(b))))

# All edges in the original graph.
original_edges = {
    edge_key(a, b)
    for a, b in graph.edges()
}

# All edges represented by extracted branches.
branch_edges = Counter()

for branch in result["branches"]:
    coordinates = branch["coordinates"]

    for i in range(len(coordinates) - 1):
        edge = edge_key(
            coordinates[i],
            coordinates[i + 1]
        )
        branch_edges[edge] += 1

covered_edges = set(branch_edges)

missing_edges = original_edges - covered_edges

extra_edges = covered_edges - original_edges

repeated_edges = {
    edge: count
    for edge, count in branch_edges.items()
    if count > 1
}

coverage = (
    len(covered_edges) / len(original_edges) * 100
    if original_edges else 0
)

print("\n========== GRAPH EDGE VALIDATION ==========")
print("Original graph edges:", len(original_edges))
print("Extracted branch edges:", sum(branch_edges.values()))
print("Unique extracted edges:", len(covered_edges))
print("Missing edges:", len(missing_edges))
print("Extra edges:", len(extra_edges))
print("Repeated edges:", len(repeated_edges))
print("Edge coverage:", round(coverage, 2), "%")

if missing_edges:
    print("\nSample missing edges:")
    for edge in list(missing_edges)[:10]:
        print(edge)

if repeated_edges:
    print("\nSample repeated edges:")
    for edge, count in list(repeated_edges.items())[:10]:
        print(edge, "Occurrences:", count)

if extra_edges:
    print("\nSample extra edges:")
    for edge in list(extra_edges)[:10]:
        print(edge)
