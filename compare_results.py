"""
compare_results.py

Optionally reruns Gurobi and/or DP solvers, then reads their CSV result files
and prints a comparative summary table.

Usage:
    python compare_results.py                        # compare existing CSVs
    python compare_results.py --rerun-gurobi         # rerun Gurobi, then compare
    python compare_results.py --rerun-dp             # rerun DP, then compare
    python compare_results.py --rerun-gurobi --rerun-dp  # rerun both
"""

import argparse
import re
import subprocess
import sys

import pandas as pd


GUROBI_CSV = "results_gurobi.csv"
DP_CSV     = "results_dp.csv"


# ---------------------------------------------------------------------------
# Optional reruns
# ---------------------------------------------------------------------------

def rerun(script):
    print(f"\n{'='*60}")
    print(f"Running {script} ...")
    print('='*60)
    result = subprocess.run([sys.executable, script], check=True)
    return result.returncode


# ---------------------------------------------------------------------------
# Load & merge
# ---------------------------------------------------------------------------

def extract_n(filename):
    """Extract item count n from instance filename."""
    m = re.search(r"_objs_(\d+)_", filename)
    return int(m.group(1)) if m else None


def load_results():
    gurobi = pd.read_csv(GUROBI_CSV)
    dp     = pd.read_csv(DP_CSV)

    gurobi["n"] = gurobi["instance_file"].apply(extract_n)
    dp["n"]     = dp["instance_file"].apply(extract_n)

    merged = gurobi.merge(
        dp[["corr_type", "instance_file", "obj_value", "runtime"]],
        on=["corr_type", "instance_file"],
        suffixes=("_gurobi", "_dp"),
    )

    merged["obj_value_gurobi"] = pd.to_numeric(merged["obj_value_gurobi"], errors="coerce")
    merged["obj_value_dp"]     = pd.to_numeric(merged["obj_value_dp"],     errors="coerce")
    merged["runtime_gurobi"]   = pd.to_numeric(merged["runtime_gurobi"],   errors="coerce")
    merged["runtime_dp"]       = pd.to_numeric(merged["runtime_dp"],       errors="coerce")

    # Relative gap: (gurobi - dp) / gurobi  (positive = gurobi found better)
    merged["obj_gap_pct"] = (
        (merged["obj_value_gurobi"] - merged["obj_value_dp"])
        / merged["obj_value_gurobi"].abs()
        * 100
    )

    return merged


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_separator(width=72):
    print("-" * width)


def summarise(merged):
    print("\n" + "="*72)
    print("COMPARISON SUMMARY: Gurobi MIP  vs  Dynamic Programming")
    print("="*72)

    # ------------------------------------------------------------------
    # Overall
    # ------------------------------------------------------------------
    print("\n--- Overall ---")
    print(f"  Instances compared : {len(merged)}")
    print(f"  Gurobi mean obj    : {merged['obj_value_gurobi'].mean():.2f}")
    print(f"  DP     mean obj    : {merged['obj_value_dp'].mean():.2f}")
    print(f"  Mean obj gap       : {merged['obj_gap_pct'].mean():.4f}%")
    print(f"  Gurobi mean time   : {merged['runtime_gurobi'].mean():.2f}s")
    print(f"  DP     mean time   : {merged['runtime_dp'].mean():.2f}s")

    # ------------------------------------------------------------------
    # By correlation type
    # ------------------------------------------------------------------
    print("\n--- By correlation type ---")
    print_separator()
    print(f"{'corr_type':<20} {'n_inst':>6} {'obj_grb':>10} {'obj_dp':>10} "
          f"{'gap%':>8} {'t_grb(s)':>10} {'t_dp(s)':>10}")
    print_separator()

    for corr, grp in merged.groupby("corr_type"):
        print(
            f"{corr:<20} {len(grp):>6} "
            f"{grp['obj_value_gurobi'].mean():>10.2f} "
            f"{grp['obj_value_dp'].mean():>10.2f} "
            f"{grp['obj_gap_pct'].mean():>8.4f} "
            f"{grp['runtime_gurobi'].mean():>10.2f} "
            f"{grp['runtime_dp'].mean():>10.2f}"
        )
    print_separator()

    # ------------------------------------------------------------------
    # By n (item count)
    # ------------------------------------------------------------------
    print("\n--- By number of items (n) ---")
    print_separator()
    print(f"{'n':>6} {'n_inst':>6} {'obj_grb':>10} {'obj_dp':>10} "
          f"{'gap%':>8} {'t_grb(s)':>10} {'t_dp(s)':>10}")
    print_separator()

    for n, grp in merged.groupby("n"):
        print(
            f"{n:>6} {len(grp):>6} "
            f"{grp['obj_value_gurobi'].mean():>10.2f} "
            f"{grp['obj_value_dp'].mean():>10.2f} "
            f"{grp['obj_gap_pct'].mean():>8.4f} "
            f"{grp['runtime_gurobi'].mean():>10.2f} "
            f"{grp['runtime_dp'].mean():>10.2f}"
        )
    print_separator()

    # ------------------------------------------------------------------
    # By corr_type x n
    # ------------------------------------------------------------------
    print("\n--- By correlation type × n ---")
    print_separator()
    print(f"{'corr_type':<20} {'n':>6} {'n_inst':>6} {'obj_grb':>10} "
          f"{'obj_dp':>10} {'gap%':>8} {'t_grb(s)':>10} {'t_dp(s)':>10}")
    print_separator()

    for (corr, n), grp in merged.groupby(["corr_type", "n"]):
        print(
            f"{corr:<20} {n:>6} {len(grp):>6} "
            f"{grp['obj_value_gurobi'].mean():>10.2f} "
            f"{grp['obj_value_dp'].mean():>10.2f} "
            f"{grp['obj_gap_pct'].mean():>8.4f} "
            f"{grp['runtime_gurobi'].mean():>10.2f} "
            f"{grp['runtime_dp'].mean():>10.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Gurobi and DP results.")
    parser.add_argument("--rerun-gurobi", action="store_true",
                        help="Re-solve all instances with Gurobi before comparing.")
    parser.add_argument("--rerun-dp", action="store_true",
                        help="Re-solve all instances with the DP before comparing.")
    args = parser.parse_args()

    if args.rerun_gurobi:
        rerun("solve_all_instances_gurobi.py")
    if args.rerun_dp:
        rerun("solve_all_instances_dp.py")

    merged = load_results()
    summarise(merged)


if __name__ == "__main__":
    main()
