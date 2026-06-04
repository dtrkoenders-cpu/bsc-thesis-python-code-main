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
    profits = np.array(lines[1].split(), dtype=int)
    weights = np.array(lines[2].split(), dtype=int)

    forfeit_sets = []
    idx = 3
    for _ in range(nS):
        nA, fC, _ = map(int, lines[idx].split())   # allowance, forfeit cost, cardinality
        items = np.array(lines[idx + 1].split(), dtype=int)
        forfeit_sets.append({'h': nA, 'd': fC, 'items': items})
        idx += 2

    return nI, nS, kS, profits, weights, forfeit_sets, k


# ---------------------------------------------------------------------------
# Proposition 2.2 — tighter upper bound on v_i
# ---------------------------------------------------------------------------

def compute_v_upper_bounds(forfeit_sets, profits):
    """
    For each forfeit set i:
      C_bar_i  = h_i highest-profit items in C_i
      C_hat_i  = items in C_i \\ C_bar_i whose profit exceeds d_i
      v_i <= |C_hat_i|

    Intuition: it only makes sense to 'accept' a violation for item j if its
    profit p_j > d_i. The top-h_i items are always taken without penalty, so 
    only the remainder is relevant.
    """
    v_ubs = []
    for fs in forfeit_sets:
        h, d, items = fs['h'], fs['d'], fs['items']
        order = np.argsort(profits[items])[::-1]   # indices into items, highest profit first
        remainder = items[order[h:]]
        v_ubs.append(int(np.sum(profits[remainder] > d)))
    return v_ubs


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpfs(instance_path):
    """
    Solve a KPFS instance.

    Returns a dict with keys:
        status, obj_value, mip_gap, runtime,
        selected_items, violations_per_set, total_violations
    """
    nI, nS, b, profits, weights, forfeit_sets, k = read_instance(instance_path)

    v_ubs = compute_v_upper_bounds(forfeit_sets, profits)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    m = gp.Model("KPFS")
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 10800)   # 3 hours
    m.setParam('MIPGap', 1e-4)

    # --- Variables ---
    x = m.addVars(nI, vtype=GRB.BINARY, name='x')
    #x = m.addVars(nI, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    v = m.addVars(nS, lb=0.0, ub=[float(ub) for ub in v_ubs],
                  vtype=GRB.CONTINUOUS, name='v')

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
            gp.quicksum(x[j] for j in fs['items']) - v[i] <= fs['h'],
            name=f'forfeit_{i}'
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    total_vio = round(sum(v[i].X for i in range(nS))) if m.SolCount > 0 else None
    return {
        'obj_value':        m.ObjVal if m.SolCount > 0 else None,
        'total_violations': total_vio,
        'runtime':          runtime,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python solve_kpfs_instance_gurobi.py <instance_path>")
        sys.exit(1)

    result = solve_kpfs(sys.argv[1])

    print(f"Runtime:          {result['runtime']:.2f}s")
    if result['obj_value'] is None:
        print("No feasible solution found.")
    else:
        print(f"Objective value:  {result['obj_value']:.4f}")


if __name__ == '__main__':
    main()
