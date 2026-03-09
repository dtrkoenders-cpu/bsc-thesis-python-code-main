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

def compute_A(item_profits, item_weights, capacity, set_size):
    """
    A[W, s] = max total profit from selecting exactly s items with total
    weight exactly W. Shape: (capacity+1, set_size+1).
    Unreachable (W, s) pairs are -inf.

    Vectorized version of:
    for W in range(capacity, w - 1, -1):
    for s in range(set_size, 0, -1):
        if A[W - w, s - 1] != -np.inf:
            A[W, s] = max(A[W, s], A[W - w, s - 1] + p)
    """
    A = np.full((capacity + 1, set_size + 1), -np.inf)
    A[0, 0] = 0.0

    for p, w in zip(item_profits, item_weights):
        if w > capacity:
            # Item too heavy to ever be selected; skip (it still counts
            # toward set cardinality, so s indices are unaffected).
            continue
        # A[W, s] = max(A[W, s], A[W-w, s-1] + p)  for W>=w, s>=1
        A[w:, 1:] = np.maximum(A[w:, 1:], A[:capacity + 1 - w, :-1] + p)

    return A


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_kpdfs(instance_path):
    """
    Solve the KPDFS via DP
    Returns a dict with keys: obj_value, runtime.
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
    # Step 3 — Base case
    # ------------------------------------------------------------------
    # f_prev[b', k'] = max profit from processed sets, using exactly b'
    # total weight, at most k' violations.
    # Before any set is processed: only (b'=0, any k') is reachable.
    f_prev = np.full((b + 1, k + 1), -np.inf)
    f_prev[0, :] = 0.0

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 4 — Bellman equation, one forfeit set at a time
    # ------------------------------------------------------------------
    for t, fs in enumerate(all_sets):
        items = fs['items']
        h, d  = fs['h'], fs['d']
        n_items = len(items)

        # Step 1: A_t[W, s]
        A = compute_A(profits[items], weights[items], b, n_items)

        # Step 2: vio_t(s) and val_t(W, s)
        s_arr = np.arange(n_items + 1)
        vio   = np.maximum(0, s_arr - h)          # shape (n_items+1,)
        val   = A - d * vio[np.newaxis, :]        # shape (b+1, n_items+1)

        # Step 4: accumulate Bellman updates into f_curr
        f_curr = np.full((b + 1, k + 1), -np.inf)

        # Iterate over valid (W, s) pairs; use numpy slicing for (b', k').
        for W, s in np.argwhere(np.isfinite(val)):
            W, s = int(W), int(s)
            v    = int(vio[s])
            if v > k:
                continue   # no valid k' can absorb this many violations
            # f_curr[b', k'] = max(..., f_prev[b'-W, k'-v] + val[W, s])
            f_curr[W:, v:] = np.maximum(f_curr[W:, v:], f_prev[:b + 1 - W, :k + 1 - v] + val[W, s])

        f_prev = f_curr
        elapsed = time.time() - t0
        print(
            f"  Set {t + 1:>{len(str(l))}}/{l}  "
            f"({n_items:>4} items, h={h}, d={d}, k={k})  "
            f"{elapsed:.1f}s elapsed"
        )

    # ------------------------------------------------------------------
    # Step 5 — Final answer
    # ------------------------------------------------------------------
    # f_prev[b', k] is the best profit achievable with exactly b' weight
    # and at most k violations. The optimum is the best across b'=0..b.
    opt     = f_prev[:b + 1, k].max()
    runtime = time.time() - t0

    return {
        'obj_value': float(opt) if np.isfinite(opt) else None,
        'runtime':   runtime,
    }


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
