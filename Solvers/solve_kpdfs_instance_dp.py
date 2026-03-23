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

def compute_A(item_profits, item_weights, capacity, set_size, heur_A="none", track_choices=False):
    """
    A[W, s] = max total profit from selecting exactly s items with total
    weight exactly W. Shape: (capacity+1, set_size+1).
    Unreachable (W, s) pairs are -inf.

    heur_A : "none"   — exact DP (default)
             "greedy" — rank by profit/weight ratio; only one (W,s) per s
             "dp"     — 1-D knapsack snapshots over ratio-sorted items;
                        A[:,s] = best profit using at most s top-ratio items

    When track_choices=True:
      "none"/"greedy" — also returns choice array of shape
                        (n_items, capacity+1, set_size+1)
      "dp"            — also returns (choice, order) where
                        choice has shape (n_items, capacity+1) and
                        order is the argsort giving the ratio ranking
    """
    n_items = len(item_profits)
    A = np.full((capacity + 1, set_size + 1), -np.inf)
    A[0, 0] = 0.0

    if heur_A == "greedy":
        # Greedy heuristic: rank items by profit/weight ratio (desc).
        # For each count s, record the cumulative (weight, profit) of the
        # top-s items. Zero-weight items are treated as infinite ratio.
        if track_choices:
            choice = np.zeros((n_items, capacity + 1, set_size + 1), dtype=bool)
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
            A[cumW, s] = cumP
            if track_choices:
                choice[t, cumW, s] = True

        if track_choices:
            return A, choice

    elif heur_A == "dp":
        # DP heuristic: sort items by profit/weight ratio (desc), then run a
        # standard 1-D 0/1 knapsack over them one item at a time.  After
        # adding the t-th sorted item (t = 0-based), snapshot the current dp
        # vector into column t+1 of A.  So A[:,s] holds the best profit
        # achievable using at most s of the top-s items by ratio at each weight.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(item_weights > 0,
                              item_profits / item_weights, np.inf)
        order = np.argsort(-ratios, kind='stable')

        dp = np.full(capacity + 1, -np.inf)
        dp[0] = 0.0

        if track_choices:
            choice = np.zeros((n_items, capacity + 1), dtype=bool)

        for t in range(n_items):
            w_t = int(item_weights[order[t]])
            p_t = float(item_profits[order[t]])
            if w_t <= capacity:
                dp_prev = dp.copy()
                candidate = dp_prev[:capacity + 1 - w_t] + p_t
                if track_choices:
                    mask = candidate > dp[w_t:]
                    choice[t, w_t:] = mask
                    dp[w_t:] = np.where(mask, candidate, dp[w_t:])
                else:
                    dp[w_t:] = np.maximum(dp[w_t:], candidate)
            A[:, t + 1] = dp

        if track_choices:
            return A, choice, order

    else:  # "none" — exact DP
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


def backtrack_set_dp(choice, order, item_indices, item_weights, W, s):
    """
    Recover which items were selected using the 'dp' heuristic choice table.
    choice[t, W] = True means sorted item t improved dp[W] at step t.
    Traces back through sorted steps 0..s-1 from final weight W.
    Returns a list of original item indices.
    """
    selected = []
    for t in range(s - 1, -1, -1):
        if W <= 0:
            break
        if choice[t, W]:
            selected.append(int(item_indices[order[t]]))
            W -= int(item_weights[order[t]])
    return selected


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpdfs(instance_path, return_items=False, heur_A="none"):
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
        A_orders       = []   # ratio sort order (only used for heur_A="dp")
        outer_W_choice = []   # outer_W_choice[t][b', k'] = W used for set t
        outer_s_choice = []   # outer_s_choice[t][b', k'] = s used for set t

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
            if heur_A == "dp":
                A, choice_A, order_A = compute_A(profits[items], weights[items], b, n_items, heur_A, track_choices=True)
                A_orders.append(order_A)
            else:
                A, choice_A = compute_A(profits[items], weights[items], b, n_items, heur_A, track_choices=True)
            A_choices.append(choice_A)
            W_ch = np.full((b + 1, k + 1), -1, dtype=int)
            s_ch = np.full((b + 1, k + 1), -1, dtype=int)
        else:
            A = compute_A(profits[items], weights[items], b, n_items, heur_A)
        total_time_A += time.time() - t_A_start

        s_arr = np.arange(n_items + 1)
        vio   = np.maximum(0, s_arr - h)
        val   = A - d * vio[np.newaxis, :]

        f_curr = np.full((b + 1, k + 1), -np.inf)

        t_bell_start = time.time()
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
        total_time_bellman += time.time() - t_bell_start

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
        'obj_value':     float(opt) if np.isfinite(opt) else None,
        'runtime':       runtime,
        'time_A':        total_time_A,
        'time_bellman':  total_time_bellman,
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

            if heur_A == "dp":
                set_selected = backtrack_set_dp(
                    A_choices[t], A_orders[t], fs['items'], weights[fs['items']], W_t, s_t
                )
            else:
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
    if len(sys.argv) not in (2, 3):
        print("Usage: python solve_kpdfs_instance_dp.py <instance_path> [none|greedy|dp]")
        sys.exit(1)

    heur = sys.argv[2] if len(sys.argv) == 3 else "none"
    result = solve_kpdfs(sys.argv[1], heur_A=heur)

    print(f"Runtime:         {result['runtime']:.2f}s")
    if result['obj_value'] is None:
        print("No feasible solution found.")
    else:
        print(f"Objective value: {result['obj_value']:.4f}")


if __name__ == '__main__':
    main()
