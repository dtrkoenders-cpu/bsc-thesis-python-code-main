import os
import sys
import csv
import glob

import numpy as np

from solve_kpdfs_instance_dp import solve_kpdfs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "generators"))

from generate_instances_overlap import read_instance  # type: ignore

DISJOINT_DIR = os.path.join(_ROOT, "instances", "overlap_disjoint")
OVERLAP_DIR  = os.path.join(_ROOT, "instances", "overlap")
RESULTS_FILE = os.path.join(_ROOT, "results", "results_overlap_dp.csv")

FIELDNAMES = [
    "scenario", "p_overlap", "corr_type", "instance_file", "n",
    "obj_disjoint", "obj_overlap", "runtime",
]


def evaluate_on_overlap(selected_items, profits, forfeit_sets, k):
    """
    Given a set of selected item indices, compute the objective value
    under the original overlapping forfeit sets.
    Returns None if total violations exceed k.
    """
    selected_set = set(selected_items)
    profit = sum(profits[j] for j in selected_items)
    total_vio = 0
    penalty   = 0
    for items, h, d in forfeit_sets:
        count = sum(1 for j in items if j in selected_set)
        vio   = max(0, count - h)
        total_vio += vio
        penalty   += d * vio
    if total_vio > k:
        return None
    return profit - penalty


def iter_instances():
    """Yield (scenario, p_folder, corr_type, disjoint_path, overlap_path) sorted."""
    for scenario in sorted(os.listdir(DISJOINT_DIR)):
        scenario_dir = os.path.join(DISJOINT_DIR, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for p_folder in sorted(os.listdir(scenario_dir)):
            p_dir = os.path.join(scenario_dir, p_folder)
            if not os.path.isdir(p_dir):
                continue
            for corr_type in sorted(os.listdir(p_dir)):
                subdir = os.path.join(p_dir, corr_type)
                if not os.path.isdir(subdir):
                    continue
                for disjoint_path in sorted(glob.glob(os.path.join(subdir, "*.txt"))):
                    filename     = os.path.basename(disjoint_path)
                    overlap_path = os.path.join(
                        OVERLAP_DIR, scenario, p_folder, corr_type, filename
                    )
                    yield scenario, p_folder, corr_type, disjoint_path, overlap_path


def main():
    instance_list = list(iter_instances())
    total = len(instance_list)

    if total == 0:
        print(f"No instances found under '{DISJOINT_DIR}/'.")
        return

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (scenario, p_folder, corr_type, disjoint_path, overlap_path) in enumerate(instance_list, 1):
            filename = os.path.basename(disjoint_path)
            n = int(filename.split("objs_")[1].split("_")[0])
            print(f"[{i}/{total}] {scenario}/{p_folder}/{corr_type}/{filename}")

            # Solve disjointified instance with DP (recover selected items)
            result = solve_kpdfs(disjoint_path, return_items=True)

            # Evaluate those items against the original overlapping forfeit sets
            obj_overlap = None
            if result["obj_value"] is not None and "selected_items" in result:
                _, _, _, k, profits, _, forfeit_sets = read_instance(overlap_path)
                obj_overlap = evaluate_on_overlap(
                    result["selected_items"], profits, forfeit_sets, k
                )

            row = {
                "scenario":      scenario,
                "p_overlap":     p_folder,
                "corr_type":     corr_type,
                "instance_file": filename,
                "n":             n,
                "obj_disjoint":  f"{result['obj_value']:.2f}" if result["obj_value"] is not None else "",
                "obj_overlap":   f"{obj_overlap:.2f}"         if obj_overlap          is not None else "",
                "runtime":       f"{result['runtime']:.2f}",
            }
            writer.writerow(row)
            csvfile.flush()

            disj_str = f"{result['obj_value']:.2f}" if result["obj_value"] is not None else "None"
            over_str = f"{obj_overlap:.2f}"          if obj_overlap          is not None else "None"
            print(f"    obj_disjoint={disj_str}  obj_overlap={over_str}  time={result['runtime']:.1f}s")

    print(f"\nDone. Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
