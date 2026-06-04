import sys
import os
import csv
import glob
import re
import io
import contextlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout

from solve_kpdwfs_instance_gurobi import solve_kpdwfs_gurobi as solve_gurobi


_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR  = os.path.join(_SCRIPT_DIR, "..", "instances", "disjoint")
_DEFAULT_CSV  = os.path.join(_SCRIPT_DIR, "kpdwfs_results.csv")

# DP instances that exceed this limit are recorded as None (Gurobi still runs).
DP_TIME_LIMIT = 180   # seconds

FIELDNAMES = [
    "filename", "scenario", "type", "n", "instance",
    "dp_obj", "dp_time", "gurobi_obj", "gurobi_time", "obj_match",
]

_PATTERN = re.compile(
    r"kpdwfs_sc(\d+)_id(\d+)_n(\d+)_b\d+_sets\d+_k\d+_([\w-]+)\.txt$"
)

_TYPE_ABBREV = {
    "not-correlated":   "NC",
    "correlated":       "C",
    "fully-correlated": "FC",
}


def discover_instances(root):
    """Return list of (filepath, meta_dict) for every kpdwfs_*.txt file under root."""
    paths = sorted(glob.glob(os.path.join(root, "**", "kpdwfs_*.txt"), recursive=True))
    results = []
    for path in paths:
        filename = os.path.basename(path)
        m = _PATTERN.match(filename)
        if m is None:
            continue
        scenario, inst_idx, n, corr = m.group(1), m.group(2), m.group(3), m.group(4)
        results.append((path, {
            "filename": filename,
            "scenario": int(scenario),
            "type":     _TYPE_ABBREV.get(corr, corr),
            "n":        int(n),
            "instance": int(inst_idx),
        }))
    return results


def _dp_worker(path):
    """Runs in a separate process so it can be killed on timeout."""
    from solve_kpdwfs_instance_dp import solve_kpdwfs as solve_dp
    with contextlib.redirect_stdout(io.StringIO()):
        return solve_dp(path, return_items=False, heur_A="none")


def run_dp(path):
    """Run DP in a subprocess; return None if it exceeds DP_TIME_LIMIT."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as exe:
        future = exe.submit(_dp_worker, path)
        try:
            return future.result(timeout=DP_TIME_LIMIT)
        except FuturesTimeout:
            return None


def run_gurobi(path):
    return solve_gurobi(path, return_items=False)


def main():
    instances_dir = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DIR
    output_csv    = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_CSV

    instance_list = [(p, m) for p, m in discover_instances(instances_dir)
                     if m["scenario"] == 4]
    total = len(instance_list)

    if total == 0:
        print(f"No instances found under '{instances_dir}'.")
        sys.exit(1)

    mismatches   = 0
    dp_times     = []
    gurobi_times = []

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, delimiter=";")
        if not file_exists:
            writer.writeheader()

        for idx, (path, meta) in enumerate(instance_list, 1):
            short = f"sc{meta['scenario']}_{meta['type']}_n{meta['n']}_inst{meta['instance']:02d}"

            dp_obj = dp_time = gurobi_obj = gurobi_time = None

            try:
                r = run_dp(path)
                if r is None:
                    print(f"WARNING: DP timed out (>{DP_TIME_LIMIT}s) on {meta['filename']}",
                          file=sys.stderr)
                else:
                    dp_obj  = r["obj_value"]
                    dp_time = r["runtime"]
            except Exception as exc:
                print(f"WARNING: DP failed on {meta['filename']}: {exc}", file=sys.stderr)

            try:
                r = run_gurobi(path)
                gurobi_obj  = r["obj_value"]
                gurobi_time = r["runtime"]
            except Exception as exc:
                print(f"WARNING: Gurobi failed on {meta['filename']}: {exc}", file=sys.stderr)

            if dp_obj is not None and gurobi_obj is not None:
                obj_match = abs(dp_obj - gurobi_obj) <= 1e-4
            else:
                obj_match = dp_obj is None and gurobi_obj is not None  # timeout is not a mismatch

            if dp_obj is not None and gurobi_obj is not None and not obj_match:
                mismatches += 1
                print(
                    f"WARNING: objective mismatch on {meta['filename']}: "
                    f"DP={dp_obj}, Gurobi={gurobi_obj}",
                    file=sys.stderr,
                )

            row = {**meta,
                   "dp_obj":      f"{dp_obj:.2f}".replace(".", ",")     if dp_obj     is not None else "",
                   "dp_time":     f"{dp_time:.4f}".replace(".", ",")    if dp_time    is not None else "",
                   "gurobi_obj":  f"{gurobi_obj:.2f}".replace(".", ",") if gurobi_obj is not None else "",
                   "gurobi_time": f"{gurobi_time:.4f}".replace(".", ",")if gurobi_time is not None else "",
                   "obj_match":   obj_match}
            writer.writerow(row)
            csvfile.flush()

            if dp_time     is not None: dp_times.append(dp_time)
            if gurobi_time is not None: gurobi_times.append(gurobi_time)

            dp_str     = (f"{dp_obj:.2f} ({dp_time:.2f}s)" if dp_obj is not None
                          else f"TIMEOUT (>{DP_TIME_LIMIT}s)" if r is None else "FAILED")
            gurobi_str = f"{gurobi_obj:.2f} ({gurobi_time:.2f}s)" if gurobi_obj is not None else "FAILED"
            status     = "OK" if obj_match else "MISMATCH"
            print(f"[{idx:>4}/{total}] {short} | DP: {dp_str} | Gurobi: {gurobi_str} | {status}")

    dp_mean     = sum(dp_times)     / len(dp_times)     if dp_times     else float("nan")
    gurobi_mean = sum(gurobi_times) / len(gurobi_times) if gurobi_times else float("nan")

    print(f"\nTotal instances:  {total}")
    print(f"Mismatches:       {mismatches}")
    print(f"DP mean time:     {dp_mean:.3f}s  (timed-out instances excluded)")
    print(f"Gurobi mean time: {gurobi_mean:.3f}s")
    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    main()
