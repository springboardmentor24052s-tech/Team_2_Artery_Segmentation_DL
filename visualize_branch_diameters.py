
import json
from pathlib import Path

import matplotlib.pyplot as plt


json_path = Path(
    "backend/results/branch_diameter_test.json"
)

with open(json_path, "r", encoding="utf-8") as file:
    result = json.load(file)

output_dir = Path("backend/results/branch_diameter_plots")
output_dir.mkdir(parents=True, exist_ok=True)

for branch in result["branches"]:

    profile = branch["diameter_profile"]

    if len(profile) < 5:
        continue

    positions = list(range(len(profile)))

    diameters = [
        point["diameter_pixels"]
        for point in profile
    ]

    plt.figure(figsize=(9, 4))

    plt.plot(
        positions,
        diameters,
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    plt.title(
        f"Branch {branch['branch_id']} Diameter Profile"
    )

    plt.xlabel("Position along ordered branch (pixels)")
    plt.ylabel("Estimated diameter (pixels)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = (
        output_dir /
        f"branch_{branch['branch_id']}_diameter.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

print("Diameter plots saved to:", output_dir)
print("Branches plotted:", sum(
    len(b["diameter_profile"]) >= 5
    for b in result["branches"]
))
