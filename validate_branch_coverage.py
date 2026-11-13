
import json
from pathlib import Path
from collections import Counter

json_path = Path(
    "backend/results/branch_geometry_test.json"
)

with open(json_path, "r", encoding="utf-8") as file:
    result = json.load(file)

branches = result["branches"]

# Count how often each pixel occurs across branches.
pixel_counts = Counter()

for branch in branches:
    for coordinate in branch["coordinates"]:
        pixel_counts[tuple(coordinate)] += 1

unique_pixels = len(pixel_counts)

repeated_pixels = sum(
    count > 1 for count in pixel_counts.values()
)

total_branch_pixels = sum(
    branch["pixel_count"] for branch in branches
)

duplicate_pairs = Counter()

for branch in branches:
    pair = tuple(sorted((
        branch["start_node"],
        branch["end_node"]
    )))
    duplicate_pairs[pair] += 1

duplicates = {
    pair: count
    for pair, count in duplicate_pairs.items()
    if count > 1
}

short_branches = [
    branch for branch in branches
    if branch["pixel_count"] < 10
]

print("\n========== BRANCH COVERAGE ==========")
print("Skeleton pixels:", result["skeleton_pixel_count"])
print("Total branch pixel entries:", total_branch_pixels)
print("Unique pixels in branches:", unique_pixels)
print("Pixels appearing in multiple branches:", repeated_pixels)

coverage = (
    unique_pixels / result["skeleton_pixel_count"] * 100
    if result["skeleton_pixel_count"] else 0
)

print("Unique-pixel coverage:", round(coverage, 2), "%")

print("\n========== DUPLICATE BRANCH CHECK ==========")
print("Repeated endpoint/junction pairs:", len(duplicates))

for pair, count in duplicates.items():
    print("Pair:", pair, "Occurrences:", count)

print("\n========== SHORT BRANCH CHECK ==========")
print("Branches shorter than 10 pixels:", len(short_branches))

for branch in short_branches:
    print(
        "Branch:", branch["branch_id"],
        "Pixels:", branch["pixel_count"]
    )
