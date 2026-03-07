import sys
import argparse
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

    nI, nS, kS = map(int, lines[0].split())
    profits = np.array(lines[1].split(), dtype=int)
    weights = np.array(lines[2].split(), dtype=int)

    forfeit_sets = []
    idx = 3
    for _ in range(nS):
        nA, fC, _ = map(int, lines[idx].split())   # allowance, forfeit cost, cardinality
        items = np.array(lines[idx + 1].split(), dtype=int)
        forfeit_sets.append({'h': nA, 'd': fC, 'items': items})
        idx += 2

    return nI, nS, kS, profits, weights, forfeit_sets


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
    profit p_j > d_i (the penalty paid per violation).  The top-h_i items are
    always taken without penalty, so only the remainder is relevant.
    """
    v_ubs = []
    for fs in forfeit_sets:
        h, d, items = fs['h'], fs['d'], fs['items']
        order = np.argsort(profits[items])[::-1]   # indices into items, highest profit first
        remainder = items[order[h:]]
        v_ubs.append(int(np.sum(profits[remainder] > d)))
    return v_ubs


# ---------------------------------------------------------------------------
# Solver callback — live elapsed-time display
# ---------------------------------------------------------------------------

def _make_timer_callback(t0):
    """Returns a Gurobi callback that prints elapsed time on a single line."""
    def callback(_, where):
        if where == GRB.Callback.MIP:
            elapsed = time.time() - t0
            print(f"\r  Solving... {elapsed:8.1f}s elapsed", end='', flush=True)
    return callback


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpfs(instance_path, verbose=False):
    """
    Solve a KPFS instance.

    Returns a dict with keys:
        status, obj_value, mip_gap, runtime,
        selected_items, violations_per_set, total_violations
    """
    nI, nS, b, profits, weights, forfeit_sets = read_instance(instance_path)

    # k = maximum total violations across all forfeit sets (not stored in file)
    k = round(nI / 15)

    v_ubs = compute_v_upper_bounds(forfeit_sets, profits)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    m = gp.Model("KPFS")
    m.setParam('OutputFlag', int(verbose))
    m.setParam('TimeLimit', 10800)   # 3 hours
    m.setParam('MIPGap', 1e-4)

    # --- Variables ---
    x = m.addVars(nI, vtype=GRB.BINARY, name='x')
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
    m.optimize(_make_timer_callback(t0))
    print()  # newline after the live clock
    runtime = time.time() - t0

    # ------------------------------------------------------------------
    # Extract results
    # ------------------------------------------------------------------
    status_map = {
        GRB.OPTIMAL:    'optimal',
        GRB.TIME_LIMIT: 'time_limit',
        GRB.INFEASIBLE: 'infeasible',
        GRB.INF_OR_UNBD: 'inf_or_unbounded',
    }
    status = status_map.get(m.Status, f'gurobi_status_{m.Status}')

    if m.SolCount == 0:
        return {
            'status': status,
            'obj_value': None,
            'mip_gap': None,
            'runtime': runtime,
            'selected_items': None,
            'violations_per_set': None,
            'total_violations': None,
        }

    x_vals = np.array([x[j].X for j in range(nI)])
    selected_items = np.where(x_vals > 0.5)[0]
    violations_per_set = np.array([v[i].X for i in range(nS)])
    total_violations = violations_per_set.sum()

    _check_feasibility(selected_items, violations_per_set,
                       weights, b, k, forfeit_sets)

    return {
        'status': status,
        'obj_value': m.ObjVal,
        'mip_gap': m.MIPGap,
        'runtime': runtime,
        'selected_items': selected_items,
        'violations_per_set': violations_per_set,
        'total_violations': total_violations,
    }


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def _check_feasibility(selected_items, violations_per_set,
                        weights, b, k, forfeit_sets, tol=1e-6):
    total_weight = weights[selected_items].sum()
    if total_weight > b + tol:
        raise RuntimeError(
            f"Feasibility check failed — budget: {total_weight:.2f} > {b}"
        )

    total_violations = violations_per_set.sum()
    if total_violations > k + tol:
        raise RuntimeError(
            f"Feasibility check failed — global violation bound: "
            f"{total_violations:.4f} > {k}"
        )

    selected_mask = np.zeros(weights.shape[0], dtype=bool)
    selected_mask[selected_items] = True
    for i, fs in enumerate(forfeit_sets):
        selected_in_set = selected_mask[fs['items']].sum()
        if selected_in_set - violations_per_set[i] > fs['h'] + tol:
            raise RuntimeError(
                f"Feasibility check failed — forfeit set {i}: "
                f"{selected_in_set} selected, {violations_per_set[i]:.4f} violations, "
                f"allowance {fs['h']}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Solve a KPFS instance with Gurobi.')
    parser.add_argument('instance', help='Path to instance file')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable Gurobi console output')
    args = parser.parse_args()

    result = solve_kpfs(args.instance, verbose=args.verbose)

    print(f"Status:           {result['status']}")
    print(f"Runtime:          {result['runtime']:.2f}s")

    if result['obj_value'] is None:
        print("No feasible solution found within the time limit.")
        sys.exit(1)

    print(f"Objective value:  {result['obj_value']:.4f}")
    print(f"MIP gap:          {result['mip_gap']:.2e}")
    print(f"Items selected:   {len(result['selected_items'])}")
    print(f"Total violations: {result['total_violations']:.4f}")


if __name__ == '__main__':
    main()
