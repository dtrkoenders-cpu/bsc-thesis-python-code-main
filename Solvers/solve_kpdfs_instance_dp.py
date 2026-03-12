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
        nA, fC, _ = map(int, lines[idx].split())   # allowance, forfeit cost, cardinality
        items = np.array(lines[idx + 1].split(), dtype=int)
        forfeit_sets.append({'h': nA, 'd': fC, 'items': items})
        idx += 2

    return nI, nS, kS, profits, weights, forfeit_sets, k


# ---------------------------------------------------------------------------
# Step 1 — A_i(W, s) for a single forfeit set
# ---------------------------------------------------------------------------

def compute_A(item_profits, item_weights, capacity, set_size, track_choices=False):
    """
    A[W, s] = max total profit from selecting exactly s items with total
    weight exactly W. Shape: (capacity+1, set_size+1).
    Unreachable (W, s) pairs are -inf.

    When track_choices=True, also returns a boolean choice array of shape
    (n_items, capacity+1, set_size+1) where choice[t, W, s] = True means
    item t was selected to reach state (W, s).
    """
    n_items = len(item_profits)
    A = np.full((capacity + 1, set_size + 1), -np.inf)
    A[0, 0] = 0.0

    if track_choices:
        choice = np.zeros((n_items, capacity + 1, set_size + 1), dtype=bool)

    for t, (p, w) in enumerate(zip(item_profits, item_weights)):
        if w > capacity:
            continue
        candidate = A[:capacity + 1 - w, :-1] + p
        if track_choices:
            mask = candidate > A[w:, 1:]
            choice[t, w:, 1:] = mask
            A[w:, 1:] = np.where(mask, candidate, A[w:, 1:])
        else:
            A[w:, 1:] = np.maximum(A[w:, 1:], candidate)

    if track_choices:
        return A, choice
    return A


def backtrack_set(choice, item_indices, item_weights, W, s):
    """
    Recover which items were selected from a single forfeit set.
    Traces back through the choice table from state (W, s).
    Returns a list of original item indices.
    """
    selected = []
    for t in range(choice.shape[0] - 1, -1, -1):
        if s == 0:
            break
        if choice[t, W, s]:
            selected.append(int(item_indices[t]))
            W -= item_weights[t]
            s -= 1
    return selected


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpdfs(instance_path, return_items=False):
    """
    Solve the KPDFS via DP.

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
    # Free items → extra forfeit set with h = |free|, d = 0
    # ------------------------------------------------------------------
    in_set = np.zeros(nI, dtype=bool)
    for fs in forfeit_sets:
        in_set[fs['items']] = True
    free_items = np.where(~in_set)[0]

    all_sets = list(forfeit_sets)
    if len(free_items) > 0:
        all_sets.append({'h': len(free_items), 'd': 0, 'items': free_items})

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
        outer_s_choice = []   # outer_s_choice[t][b', k'] = s used for set t

    # ------------------------------------------------------------------
    # Bellman equation, one forfeit set at a time
    # ------------------------------------------------------------------
    for t, fs in enumerate(all_sets):
        items   = fs['items']
        h, d    = fs['h'], fs['d']
        n_items = len(items)

        if return_items:
            A, choice_A = compute_A(profits[items], weights[items], b, n_items, track_choices=True)
            A_choices.append(choice_A)
            W_ch = np.full((b + 1, k + 1), -1, dtype=int)
            s_ch = np.full((b + 1, k + 1), -1, dtype=int)
        else:
            A = compute_A(profits[items], weights[items], b, n_items)

        s_arr = np.arange(n_items + 1)
        vio   = np.maximum(0, s_arr - h)
        val   = A - d * vio[np.newaxis, :]

        f_curr = np.full((b + 1, k + 1), -np.inf)

        for W_val, s_val in np.argwhere(np.isfinite(val)):
            W_val, s_val = int(W_val), int(s_val)
            v = int(vio[s_val])
            if v > k:
                continue
            candidate = f_prev[:b + 1 - W_val, :k + 1 - v] + val[W_val, s_val]
            mask = candidate > f_curr[W_val:, v:]
            if return_items:
                W_ch[W_val:, v:][mask] = W_val
                s_ch[W_val:, v:][mask] = s_val
            f_curr[W_val:, v:] = np.where(mask, candidate, f_curr[W_val:, v:])

        if return_items:
            outer_W_choice.append(W_ch)
            outer_s_choice.append(s_ch)

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
        'obj_value': float(opt) if np.isfinite(opt) else None,
        'runtime':   runtime,
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
            s_t = int(outer_s_choice[t][b_curr, k_curr])
            fs  = all_sets[t]
            v_t = int(max(0, s_t - fs['h']))

            set_selected = backtrack_set(
                A_choices[t], fs['items'], weights[fs['items']], W_t, s_t
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
    if len(sys.argv) != 2:
        print("Usage: python solve_kpdfs_instance_dp.py <instance_path>")
        sys.exit(1)

    result = solve_kpdfs(sys.argv[1])

    print(f"Runtime:         {result['runtime']:.2f}s")
    if result['obj_value'] is None:
        print("No feasible solution found.")
    else:
        print(f"Objective value: {result['obj_value']:.4f}")


if __name__ == '__main__':
    main()
