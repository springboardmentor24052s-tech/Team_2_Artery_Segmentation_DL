
import json
from pathlib import Path
from collections import defaultdict

path = Path(
    "backend/results/branch_diameter_test.json"
)

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

node_to_branches = defaultdict(list)

for branch in data["branches"]:
    branch_id = branch["branch_id"]

    node_to_branches[branch["start_node"]].append(branch_id)
    node_to_branches[branch["end_node"]].append(branch_id)

print("\n========== BRANCH CONNECTIVITY REPORT ==========")

for node, branch_ids in sorted(node_to_branches.items()):

    if len(branch_ids) > 1:
        print(
            f"Node {node}: "
            f"connected branches {sorted(branch_ids)}"
        )

print("\n========== CANDIDATE BRANCH CONNECTIVITY ==========")

candidate_path = Path(
    "backend/results/narrowing_candidate_analysis.json"
)

with open(candidate_path, "r", encoding="utf-8") as file:
    candidates = json.load(file)

for candidate in candidates:

    branch_id = candidate["branch_id"]

    branch = next(
        b for b in data["branches"]
        if b["branch_id"] == branch_id
    )

    start = branch["start_node"]
    end = branch["end_node"]

    print(
        f"\nBranch {branch_id}: "
        f"start={start}, end={end}"
    )

    print(
        "Branches connected at start:",
        sorted(node_to_branches[start])
    )

    print(
        "Branches connected at end:",
        sorted(node_to_branches[end])
    )
