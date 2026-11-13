
import sys
import cv2
import networkx as nx
import numpy as np
from collections import Counter

sys.path.insert(0, "backend")

from app.services.branch_geometry import build_skeleton_graph

IMAGE_ID = "16c135650bb24194a3abfd836c03237c"

mask_path = (
    f"backend/results/api_predictions/"
    f"{IMAGE_ID}_cleaned_mask.png"
)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(mask_path)

from app.services.centerline import extract_centerline

centerline = extract_centerline(mask)["centerline_mask"]

graph = build_skeleton_graph(centerline)

degrees = dict(graph.degree())
degree_counts = Counter(degrees.values())

critical_pixels = {
    node for node, degree in degrees.items()
    if degree != 2
}

critical_graph = graph.subgraph(critical_pixels)
critical_regions = list(nx.connected_components(critical_graph))

print("\n========== GRAPH DIAGNOSTICS ==========")
print("Total graph nodes:", graph.number_of_nodes())
print("Total graph edges:", graph.number_of_edges())
print("Connected components:", nx.number_connected_components(graph))

print("\nPixel degree distribution:")
for degree, count in sorted(degree_counts.items()):
    print(f"Degree {degree}: {count} pixels")

print("\nCritical pixel count:", len(critical_pixels))
print("Critical region count:", len(critical_regions))

print("\nCritical region sizes:")
for i, region in enumerate(
    sorted(critical_regions, key=len, reverse=True), start=1
):
    print(f"Region {i}: {len(region)} pixels")
