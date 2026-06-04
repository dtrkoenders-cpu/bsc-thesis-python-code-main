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
import csv
import os
import sys
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

VERBOSE_CALLBACK = True
LOG_INTERVAL_S   = 3.0   # seconds between periodic progress snapshots


def build_callback():
    log = []
    last_log_t = [0.0]   # mutable so the closure can update it

    def cb(model, where):
        if where == GRB.Callback.MIP:
            t          = model.cbGet(GRB.Callback.RUNTIME)
            if t - last_log_t[0] < LOG_INTERVAL_S:
                return
            last_log_t[0] = t
            obj        = model.cbGet(GRB.Callback.MIP_OBJBST)
            bnd        = model.cbGet(GRB.Callback.MIP_OBJBND)
            node_count = int(model.cbGet(GRB.Callback.MIP_NODCNT))
            if obj < GRB.INFINITY and bnd < GRB.INFINITY:
                gap_pct = abs(bnd - obj) / (1e-10 + abs(obj)) * 100
            else:
                gap_pct = None
            log.append(('progress', round(t, 4), obj, bnd, gap_pct, node_count))

        elif where == GRB.Callback.MIPSOL:
            t   = model.cbGet(GRB.Callback.RUNTIME)
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if obj < GRB.INFINITY and bnd < GRB.INFINITY:
                gap_pct = abs(bnd - obj) / (1e-10 + abs(obj)) * 100
            else:
                gap_pct = None
            log.append(('incumbent', round(t, 4), obj, bnd, gap_pct, None))
            if VERBOSE_CALLBACK:
                gap_str = f"{gap_pct:.4f}%" if gap_pct is not None else "inf"
                print(f"[t={t:.2f}s] New incumbent: {obj:.4f}  bound: {bnd:.4f}  gap: {gap_str}")

        elif where == GRB.Callback.MIPNODE:
            node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
            status     = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if node_count == 0 and status == GRB.OPTIMAL:
                t   = model.cbGet(GRB.Callback.RUNTIME)
                bnd = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                log.append(('root_lp', round(t, 4), None, bnd, None, None))
                if VERBOSE_CALLBACK:
                    print(f"[root] LP relaxation bound: {bnd:.4f}")

    return cb, log


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------

def _write_log(solve_log, instance_path):
    if not solve_log:
        return
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(instance_path))[0]
    log_path = os.path.join(log_dir, stem + '_gurobi_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['event', 'runtime_s', 'obj', 'bound', 'gap_pct', 'node_count'])
        writer.writerows(solve_log)
    print(f"Solve log written to: {log_path}")


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
    cb, solve_log = build_callback()
    t0 = time.time()
    try:
        m.optimize(cb)
    except gp.GurobiError as e:
        runtime = time.time() - t0
        print(f"Gurobi error after {runtime:.2f}s: {e}")
        _write_log(solve_log, instance_path)
        raise

    runtime = time.time() - t0

    _write_log(solve_log, instance_path)

    result = {
        'obj_value': m.ObjVal if m.SolCount > 0 else None,
        'runtime':   runtime,
        'solve_log': solve_log,
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
