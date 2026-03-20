import os
import sys
import csv
import glob

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Solvers"))
sys.path.insert(0, os.path.join(_ROOT, "generators"))

from solve_kpdfs_instance_dp import solve_kpdfs          # type: ignore
from solve_kpfs_instance_gurobi import solve_kpfs        # type: ignore
from generate_instances_overlap import read_instance     # type: ignore

# ---------------------------------------------------------------------------
# Subset configuration
# ---------------------------------------------------------------------------
N_SUBSET      = {300}   # only run on instances with this many items
MAX_PER_GROUP = 3       # at most this many instances per (scenario/p/corr) group

DISJOINT_DIR = os.path.join(_ROOT, "instances", "overlap_disjoint")
OVERLAP_DIR  = os.path.join(_ROOT, "instances", "overlap")
RESULTS_FILE = os.path.join(_ROOT, "test", "test_results.csv")

FIELDNAMES = [
    "scenario", "p_overlap", "corr_type", "instance_file", "n",
    "k", "violations", "new_sets", 
    "obj_disjoint", "obj_overlap", "runtime_dp",
    "obj_gurobi", "violations_gurobi", "runtime_gurobi",
]


def evaluate_on_overlap(selected_items, profits, forfeit_sets, k):
    """Returns (obj, total_vio), or (None, total_vio) if violations exceed k."""
    selected_set = set(selected_items)
    profit    = sum(profits[j] for j in selected_items)
    total_vio = 0
    penalty   = 0
    for items, h, d in forfeit_sets:
        count     = sum(1 for j in items if j in selected_set)
        vio       = max(0, count - h)
        total_vio += vio
        penalty   += d * vio
    if total_vio > k:
        return None, total_vio
    return profit - penalty, total_vio


def iter_instances():
    """Yield (scenario, p_folder, corr_type, disjoint_path, overlap_path),
    filtered to N_SUBSET and at most MAX_PER_GROUP per group."""
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
                count = 0
                for disjoint_path in sorted(glob.glob(os.path.join(subdir, "*.txt"))):
                    filename = os.path.basename(disjoint_path)
                    n = int(filename.split("objs_")[1].split("_")[0])
                    if n not in N_SUBSET:
                        continue
                    if count >= MAX_PER_GROUP:
                        break
                    overlap_path = os.path.join(
                        OVERLAP_DIR, scenario, p_folder, corr_type, filename
                    )
                    yield scenario, p_folder, corr_type, disjoint_path, overlap_path
                    count += 1


def main():
    instance_list = list(iter_instances())
    total = len(instance_list)

    if total == 0:
        print(f"No instances found. Check N_SUBSET={N_SUBSET} and '{DISJOINT_DIR}'.")
        return

    print(f"Running on {total} instances (n in {N_SUBSET}, max {MAX_PER_GROUP} per group)\n")

    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (scenario, p_folder, corr_type, disjoint_path, overlap_path) in enumerate(instance_list, 1):
            filename = os.path.basename(disjoint_path)
            n = int(filename.split("objs_")[1].split("_")[0])
            print(f"[{i}/{total}] {scenario}/{p_folder}/{corr_type}/{filename}")

            # --- DP on disjoint instance ---
            result_dp = solve_kpdfs(disjoint_path, return_items=True, heur_A=True)

            _, l_overlap,  _, k, profits, _, forfeit_sets = read_instance(overlap_path)
            _, l_disjoint, _, _, _,       _, _            = read_instance(disjoint_path)
            new_sets = l_disjoint - l_overlap

            obj_overlap = None
            total_vio   = None
            if result_dp["obj_value"] is not None and "selected_items" in result_dp:
                obj_overlap, total_vio = evaluate_on_overlap(
                    result_dp["selected_items"], profits, forfeit_sets, k
                )

            # --- Gurobi on original overlapping instance ---
            result_gurobi = solve_kpfs(overlap_path)

            row = {
                "scenario":      scenario,
                "p_overlap":     p_folder,
                "corr_type":     corr_type,
                "instance_file": filename,
                "n":             n,
                "k":             k,
                "violations":    total_vio if total_vio is not None else "",
                "new_sets":      new_sets,
                "obj_disjoint":  f"{result_dp['obj_value']:.2f}" if result_dp["obj_value"] is not None else "",
                "obj_overlap":   f"{obj_overlap:.2f}"            if obj_overlap             is not None else "",
                "runtime_dp":    f"{result_dp['runtime']:.2f}",
                "obj_gurobi":         f"{result_gurobi['obj_value']:.2f}" if result_gurobi["obj_value"] is not None else "",
                "violations_gurobi":  result_gurobi["total_violations"] if result_gurobi["total_violations"] is not None else "",
                "runtime_gurobi":     f"{result_gurobi['runtime']:.2f}",
            }
            writer.writerow(row)
            csvfile.flush()

            dp_str  = f"{result_dp['obj_value']:.2f}"       if result_dp["obj_value"]      is not None else "None"
            ov_str  = f"{obj_overlap:.2f}"                   if obj_overlap                 is not None else "None"
            gr_str  = f"{result_gurobi['obj_value']:.2f}"   if result_gurobi["obj_value"]  is not None else "None"
            print(f"    obj_disjoint={dp_str}  obj_overlap={ov_str}  obj_gurobi={gr_str}"
                  f"  t_dp={result_dp['runtime']:.1f}s  t_gurobi={result_gurobi['runtime']:.1f}s")

    print(f"\nDone. Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
