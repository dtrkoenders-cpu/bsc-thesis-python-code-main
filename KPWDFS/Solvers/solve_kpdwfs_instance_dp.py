"""
DP solver for the Knapsack Problem with Disjoint Weight-based Forfeit Sets
(KPDWFS). Structured as close as possible to solve_kpdfs_instance_dp.py.

Key difference from KPDFS: violations are measured in weight, not item count.
  vio_i(W) = max(0, W - h_i),  where W = sum_{j in C_i} w_j * x_j
and k bounds the total excess weight across all forfeit sets.

Consequence for the DP:
  - compute_A produces a 1-D table A[W] (exact weight), dropping the s-axis.
  - The Bellman loop iterates over W_val only (no s_val dimension).
  - Backtracking traces W only; s is not tracked.

See solve_kpdfs_instance_dp.py for the item-count reference implementation.
"""
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Instance
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
# Step 1 — A_i(W) for a single forfeit set
# ---------------------------------------------------------------------------

def compute_A(item_profits, item_weights, capacity, heur_A="none", track_choices=False):
    """
    A[W] = max total profit from items in the set with total weight exactly W.
    Shape: (capacity+1,). Unreachable weights are -inf.

    heur_A : "none"   — exact 1-D 0/1 knapsack DP (default)
             "greedy" — rank by profit/weight ratio; one entry per item added

    When track_choices=True, also returns choice array of shape (n_items, capacity+1);
    choice[t, W]=True means item t was taken to reach W.
    """
    n_items = len(item_profits)
    A = np.full(capacity + 1, -np.inf)
    A[0] = 0.0

    if heur_A == "greedy":
        # Greedy heuristic: rank items by profit/weight ratio (desc).
        # Record cumulative (weight, profit) after each item added.
        if track_choices:
            choice = np.zeros((n_items, capacity + 1), dtype=bool)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(item_weights > 0,
                              item_profits / item_weights, np.inf)
        order = np.argsort(-ratios, kind='stable')

        cumW = 0
        cumP = 0.0
        for s in range(1, n_items + 1):
            t = order[s - 1]
            cumW += int(item_weights[t])
            cumP += float(item_profits[t])
            if cumW > capacity:
                break           # all larger s are also infeasible
            A[cumW] = cumP
            if track_choices:
                choice[t, cumW] = True

        if track_choices:
            return A, choice

    else:  # "none" — exact DP
        if track_choices:
            choice = np.zeros((n_items, capacity + 1), dtype=bool)
        for t, (p, w) in enumerate(zip(item_profits, item_weights)):
            if w > capacity:
                continue
            candidate = A[:capacity + 1 - w] + p
            if track_choices:
                mask = candidate > A[w:]
                choice[t, w:] = mask
                A[w:] = np.where(mask, candidate, A[w:])
            else:
                A[w:] = np.maximum(A[w:], candidate)

        if track_choices:
            return A, choice

    return A


def backtrack_set(choice, item_indices, item_weights, W):
    """
    Recover which items were selected from a single forfeit set.
    Traces back through the choice table from target weight W.
    Returns a list of original item indices.
    """
    selected = []
    for t in range(choice.shape[0] - 1, -1, -1):
        if W == 0:
            break
        if choice[t, W]:
            selected.append(int(item_indices[t]))
            W -= int(item_weights[t])
    return selected


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpdwfs(instance_path, return_items=False, heur_A="none"):
    """
    Solve the KPDWFS via DP.

    Parameters
    ----------
    return_items : bool
        If True, also recover and return the selected item indices via
        backtracking. Requires additional memory.

    Returns
    -------
    dict with keys:
        obj_value      : float or None
        runtime        : float
        time_A         : float
        time_bellman   : float
        selected_items : list of int  (only if return_items=True and feasible)
    """
    nI, nS, b, profits, weights, forfeit_sets, k = read_instance(instance_path)

    # ------------------------------------------------------------------
    # Disjointness check
    # ------------------------------------------------------------------
    if nS > 0:
        all_items = np.concatenate([fs['items'] for fs in forfeit_sets])
        unique_items = np.unique(all_items)
        if len(unique_items) != len(all_items):
            raise ValueError(
                f"Forfeit sets are not pairwise disjoint: "
                f"{len(all_items) - len(unique_items)} item(s) appear in multiple sets."
            )

    # ------------------------------------------------------------------
    # Free items → extra forfeit set with h = total weight of free items, d = 0
    # (weight allowance equals full set weight, so violation is always 0)
    # ------------------------------------------------------------------
    in_set = np.zeros(nI, dtype=bool)
    for fs in forfeit_sets:
        in_set[fs['items']] = True
    free_items = np.where(~in_set)[0]

    all_sets = list(forfeit_sets)
    if len(free_items) > 0:
        all_sets.append({'h': int(weights[free_items].sum()), 'd': 0, 'items': free_items})

    l = len(all_sets)

    # ------------------------------------------------------------------
    # Base case
    # ------------------------------------------------------------------
    f_prev = np.full((b + 1, k + 1), -np.inf)
    f_prev[0, :] = 0.0

    t0 = time.time()

    if return_items:
        A_choices      = []   # choice tables from compute_A, one per set
        outer_W_choice = []   # outer_W_choice[t][b', k'] = W used for set t

    # ------------------------------------------------------------------
    # Bellman equation, one forfeit set at a time
    # ------------------------------------------------------------------
    total_time_A       = 0.0
    total_time_bellman = 0.0

    for t, fs in enumerate(all_sets):
        items   = fs['items']
        h, d    = fs['h'], fs['d']
        n_items = len(items)

        t_A_start = time.time()
        if return_items:
            A, choice_A = compute_A(profits[items], weights[items], b, heur_A, track_choices=True)
            A_choices.append(choice_A)
            W_ch = np.full((b + 1, k + 1), -1, dtype=int)
        else:
            A = compute_A(profits[items], weights[items], b, heur_A)
        total_time_A += time.time() - t_A_start

        W_arr = np.arange(b + 1)
        vio   = np.maximum(0, W_arr - h)
        val   = A - d * vio

        f_curr = np.full((b + 1, k + 1), -np.inf)

        t_bell_start = time.time()
        for W_val in np.where(np.isfinite(val))[0]:
            W_val = int(W_val)
            v = int(vio[W_val])
            if v > k:
                continue
            candidate = f_prev[:b + 1 - W_val, :k + 1 - v] + val[W_val]
            mask = candidate > f_curr[W_val:, v:]
            if return_items:
                W_ch[W_val:, v:][mask] = W_val
            f_curr[W_val:, v:] = np.where(mask, candidate, f_curr[W_val:, v:])
        total_time_bellman += time.time() - t_bell_start

        if return_items:
            outer_W_choice.append(W_ch)

        f_prev = f_curr
        elapsed = time.time() - t0
        print(
            f"  Set {t + 1:>{len(str(l))}}/{l}  "
            f"({n_items:>4} items, h={h}, d={d}, k={k})  "
            f"{elapsed:.1f}s elapsed"
        )

    # ------------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------------
    opt     = f_prev[:b + 1, k].max()
    runtime = time.time() - t0

    result = {
        'obj_value':    float(opt) if np.isfinite(opt) else None,
        'runtime':      runtime,
        'time_A':       total_time_A,
        'time_bellman': total_time_bellman,
    }

    # ------------------------------------------------------------------
    # Backtracking (only when requested and a solution exists)
    # ------------------------------------------------------------------
    if return_items and np.isfinite(opt):
        b_curr = int(np.argmax(f_prev[:b + 1, k]))
        k_curr = k
        selected = []

        for t in range(l - 1, -1, -1):
            W_t = int(outer_W_choice[t][b_curr, k_curr])
            fs  = all_sets[t]
            v_t = int(max(0, W_t - fs['h']))

            set_selected = backtrack_set(
                    A_choices[t], fs['items'], weights[fs['items']], W_t
                )
            selected.extend(set_selected)

            b_curr -= W_t
            k_curr -= v_t

        result['selected_items'] = sorted(selected)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python solve_kpdwfs_instance_dp.py <instance_path> [none|greedy]")
        sys.exit(1)

    heur = sys.argv[2] if len(sys.argv) == 3 else "none"
    result = solve_kpdwfs(sys.argv[1], heur_A=heur)

    print(f"Runtime:         {result['runtime']:.2f}s")
    if result['obj_value'] is None:
        print("No feasible solution found.")
    else:
        print(f"Objective value: {result['obj_value']:.4f}")


if __name__ == '__main__':
    main()
