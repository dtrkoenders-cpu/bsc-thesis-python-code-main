import os
import csv
import glob

from solve_kpfs_instance_gurobi import solve_kpfs


_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR = os.path.join(_ROOT, "instances", "overlap")
RESULTS_FILE = os.path.join(_ROOT, "results", "results_overlap_gurobi.csv")

FIELDNAMES = ["scenario", "p_overlap", "corr_type", "instance_file", "n", "obj_value", "runtime"]


def iter_instances():
    """Yield (scenario, p_overlap, corr_type, filepath) for every instance file, sorted."""
    for scenario in sorted(os.listdir(INSTANCE_DIR)):
        scenario_dir = os.path.join(INSTANCE_DIR, scenario)
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
                for path in sorted(glob.glob(os.path.join(subdir, "*.txt"))):
                    yield scenario, p_folder, corr_type, path

def main():
    instance_list = list(iter_instances())
    total = len(instance_list)

    if total == 0:
        print(f"No instances found under '{INSTANCE_DIR}/'.")
        return

    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (scenario, p_folder, corr_type, path) in enumerate(instance_list, 1):
            filename = os.path.basename(path)
            n = int(filename.split("objs_")[1].split("_")[0])
            print(f"[{i}/{total}] {scenario}/{p_folder}/{corr_type}/{filename}")

            result = solve_kpfs(path)

            row = {
                "scenario":      scenario,
                "p_overlap":     p_folder,
                "corr_type":     corr_type,
                "instance_file": filename,
                "n":             n,
                "obj_value":     result["obj_value"],
                "runtime":       f"{result['runtime']:.2f}",
            }
            writer.writerow(row)
            csvfile.flush()

            if result["obj_value"] is not None:
                print(f"    obj={result['obj_value']:.2f}  time={result['runtime']:.1f}s")
            else:
                print(f"    no solution found")

    print(f"\nDone. Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
