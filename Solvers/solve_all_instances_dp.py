import os
import csv
import glob

from solve_kpdfs_instance_dp import solve_kpdfs


_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCE_DIR = os.path.join(_ROOT, "instances", "disjoint")
RESULTS_FILE = os.path.join(_ROOT, "results", "results_dp.csv")

FIELDNAMES = ["scenario", "corr_type", "instance_file", "n", "obj_value", "runtime"]


def iter_instances():
    """Yield (scenario, corr_type, filepath) for every instance file, sorted."""
    for scenario in sorted(os.listdir(INSTANCE_DIR)):
        scenario_dir = os.path.join(INSTANCE_DIR, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for corr_type in sorted(os.listdir(scenario_dir)):
            subdir = os.path.join(scenario_dir, corr_type)
            if not os.path.isdir(subdir):
                continue
            for path in sorted(glob.glob(os.path.join(subdir, "*.txt"))):
                yield scenario, corr_type, path


def main():
    instance_list = list(iter_instances())
    total = len(instance_list)

    if total == 0:
        print(f"No instances found under '{INSTANCE_DIR}/'.")
        return

    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (scenario, corr_type, path) in enumerate(instance_list, 1):
            filename = os.path.basename(path)
            n = int(filename.split("objs_")[1].split("_")[0])
            print(f"[{i}/{total}] {scenario}/{corr_type}/{filename}")

            result = solve_kpdfs(path)

            row = {
                "scenario":      scenario,
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
                print(f"    no feasible solution found")

    print(f"\nDone. Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
