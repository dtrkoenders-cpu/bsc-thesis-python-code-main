"""
Gurobi ILP solver for the Knapsack Problem with Disjoint Weight-based Forfeit
Sets (KPDWFS).

Weight-based variant of the KPDFS ILP: violations v[i] measure excess weight
beyond the weight allowance h_i, not excess item count. The per-set constraint
therefore uses w_j as coefficients on x[j], and v[i]'s upper bound is the
total set weight minus h_i.

Reference ILP (solve_kpdfs_instance_gurobi.py / solve_kpfs_instance_gurobi.py):
    per-set: sum_{j in C_i} x[j]       - v[i] <= h_i   (item-count based)

This file:
    per-set: sum_{j in C_i} w_j * x[j] - v[i] <= h_i   (weight based)

DP counterpart: solve_kpdwfs_instance_dp.py
"""
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Instance I/O
# ---------------------------------------------------------------------------

def read_instance(path):
    with open(path) as f:
        lines = f.read().splitlines()

    nI, nS, kS, k = map(int, lines[0].split())
    profits = np.array(lines[1].split(), dtype=float)
    weights = np.array(lines[2].split(), dtype=int)

    forfeit_sets = []
    idx = 3
    for _ in range(nS):
        nA, fC, _ = map(int, lines[idx].split())   # allowance (weight), forfeit cost, cardinality
        items = np.array(lines[idx + 1].split(), dtype=int)
        forfeit_sets.append({'h': nA, 'd': fC, 'items': items})
        idx += 2

    return nI, nS, kS, profits, weights, forfeit_sets, k


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpdwfs_gurobi(instance_path, time_limit=None, return_items=False):
    """
    Solve a KPDWFS instance via Gurobi ILP.

    Parameters
    ----------
    time_limit   : float or None  — Gurobi TimeLimit in seconds (None = no limit)
    return_items : bool           — if True, include selected item indices in result

    Returns
    -------
    dict with keys:
        obj_value      : float or None
        runtime        : float
        mip_gap        : float  (present when a solution was found)
        selected_items : list of int  (only if return_items=True and feasible)
    """
    nI, nS, b, profits, weights, forfeit_sets, k = read_instance(instance_path)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    m = gp.Model("KPDWFS")
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 1e-4
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # --- Variables ---
    x = m.addVars(nI, vtype=GRB.BINARY, name='x')

    # v[i]: excess weight beyond h_i for forfeit set i.
    # Upper bound: total weight of all items in C_i minus h_i.
    v_ubs = [
        float(int(weights[fs['items']].sum()) - fs['h'])
        for fs in forfeit_sets
    ]
    v = m.addVars(nS, lb=0.0, ub=v_ubs, vtype=GRB.CONTINUOUS, name='v')

    # --- Objective ---
    m.setObjective(
        gp.quicksum(profits[j] * x[j] for j in range(nI))
        - gp.quicksum(forfeit_sets[i]['d'] * v[i] for i in range(nS)),
        GRB.MAXIMIZE
    )

    # --- Constraints ---
    m.addConstr(
        gp.quicksum(weights[j] * x[j] for j in range(nI)) <= b,
        name='budget'
    )

    m.addConstr(
        gp.quicksum(v[i] for i in range(nS)) <= k,
        name='global_violations'
    )

    for i, fs in enumerate(forfeit_sets):
        m.addConstr(
            gp.quicksum(int(weights[j]) * x[j] for j in fs['items']) - v[i] <= fs['h'],
            name=f'forfeit_{i}'
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    result = {
        'obj_value': m.ObjVal if m.SolCount > 0 else None,
        'runtime':   runtime,
    }

    if m.SolCount > 0:
        result['mip_gap'] = m.MIPGap

    if return_items and m.SolCount > 0:
        result['selected_items'] = sorted(
            j for j in range(nI) if x[j].X > 0.5
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python solve_kpdwfs_instance_gurobi.py <instance_path>")
        sys.exit(1)

    result = solve_kpdwfs_gurobi(sys.argv[1])

    print(f"Runtime:         {result['runtime']:.2f}s")
    if result['obj_value'] is None:
        print("No feasible solution found.")
    else:
        print(f"Objective value: {result['obj_value']:.4f}")
        print(f"MIP gap:         {result['mip_gap']:.6f}")


if __name__ == '__main__':
    main()