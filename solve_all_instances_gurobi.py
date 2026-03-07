import os
import csv
import glob

from solve_kpfs_instance_gurobi import solve_kpfs


INSTANCE_DIR = "instances"
RESULTS_FILE = "results_gurobi.csv"

FIELDNAMES = [
    "corr_type", "instance_file",
    "status", "obj_value", "mip_gap", "runtime",
    "n_items_selected", "total_violations",
]


def iter_instances():
    """Yield (corr_type, filepath) for every instance file, sorted."""
    for corr_type in sorted(os.listdir(INSTANCE_DIR)):
        subdir = os.path.join(INSTANCE_DIR, corr_type)
        if not os.path.isdir(subdir):
            continue
        for path in sorted(glob.glob(os.path.join(subdir, "*.txt"))):
            yield corr_type, path


def main():
    instance_list = list(iter_instances())
    total = len(instance_list)

    if total == 0:
        print(f"No instances found under '{INSTANCE_DIR}/'.")
        return

    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, (corr_type, path) in enumerate(instance_list, 1):
            filename = os.path.basename(path)
            print(f"[{i}/{total}] {corr_type}/{filename}")

            result = solve_kpfs(path, verbose=False)

            row = {
                "corr_type":        corr_type,
                "instance_file":    filename,
                "status":           result["status"],
                "obj_value":        result["obj_value"],
                "mip_gap":          f"{result['mip_gap']:.2e}" if result["mip_gap"] is not None else None,
                "runtime":          f"{result['runtime']:.2f}",
                "n_items_selected": len(result["selected_items"]) if result["selected_items"] is not None else None,
                "total_violations": f"{result['total_violations']:.4f}" if result["total_violations"] is not None else None,
            }
            writer.writerow(row)
            csvfile.flush()

            # Print a one-line summary after each solve
            if result["obj_value"] is not None:
                print(
                    f"    status={result['status']}  "
                    f"obj={result['obj_value']:.2f}  "
                    f"gap={result['mip_gap']:.2e}  "
                    f"time={result['runtime']:.1f}s  "
                    f"violations={result['total_violations']:.1f}"
                )
            else:
                print(f"    status={result['status']}  no solution found")

    print(f"\nDone. Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
