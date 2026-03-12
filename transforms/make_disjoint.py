import sys
import os
import copy
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "generators"))
from generate_instances_overlap import read_instance, write_instance  # type: ignore


def make_disjoint(n, l, b, k, profits, weights, forfeit_sets):
    """
    Transform a KPFS instance into a fully disjoint one.

    For every pair (C_a, C_b) sharing items S_ab:
      - New set C_ab : items = S_ab,  d = d_a + d_b,  h = min(h_a, h_b)
      - C_a keeps C_a \\ S_ab  with  h_a' = floor(h_a * |C_a \\ S_ab| / |C_a|)
      - C_b keeps C_b \\ S_ab  with  h_b' = floor(h_b * |C_b \\ S_ab| / |C_b|)
      - Sets that become empty are dropped.

    Parameters
    ----------
    forfeit_sets : list of (items, h, d) tuples
        Each item must appear in at most 2 sets.

    Returns
    -------
    (n, l_new, b, k, profits, weights, forfeit_sets_new)
    """
    forfeit_sets = copy.deepcopy(forfeit_sets)

    # --- 1. Map each item -> set indices ---
    item_to_sets = {j: [] for j in range(n)}
    for i, (items, h, d) in enumerate(forfeit_sets):
        for j in items:
            item_to_sets[int(j)].append(i)

    for j, sets in item_to_sets.items():
        if len(sets) > 2:
            raise ValueError(
                f"Item {j} belongs to {len(sets)} forfeit sets; "
                "make_disjoint requires at-most-2 overlap."
            )

    # --- 2. Find sharing pairs: (a, b) -> [shared item indices] ---
    shared_pairs = {}
    for j, sets in item_to_sets.items():
        if len(sets) == 2:
            a, b_idx = sorted(sets)
            if (a, b_idx) not in shared_pairs:
                shared_pairs[(a, b_idx)] = []
            shared_pairs[(a, b_idx)].append(j)

    # --- 3. Items to remove from each original set ---
    items_to_remove = {i: set() for i in range(l)}
    for (a, b_idx), items in shared_pairs.items():
        items_to_remove[a].update(items)
        items_to_remove[b_idx].update(items)

    # --- 4. Rebuild original sets ---
    result_sets = []
    for i, (items, h, d) in enumerate(forfeit_sets):
        n_orig    = len(items)
        remaining = np.array([j for j in items if j not in items_to_remove[i]])
        n_rem     = len(remaining)

        if n_rem == 0:
            continue

        h_new = int(np.floor(h * n_rem / n_orig)) if n_orig > 0 else 0
        result_sets.append((remaining, h_new, d))

    # --- 5. Add one merged set per sharing pair ---
    for (a, b_idx), shared_items in shared_pairs.items():
        _, h_a, d_a = forfeit_sets[a]
        _, h_b, d_b = forfeit_sets[b_idx]
        result_sets.append((np.array(sorted(shared_items)), min(h_a, h_b), d_a + d_b))

    # --- 6. Validate ---
    _validate_disjoint(result_sets)

    return n, len(result_sets), b, k, profits, weights, result_sets


def _validate_disjoint(forfeit_sets):
    seen = {}
    for i, (items, _, _) in enumerate(forfeit_sets):
        for j in items:
            j = int(j)
            if j in seen:
                raise ValueError(
                    f"Disjointness violated: item {j} appears in set {seen[j]} and set {i}."
                )
            seen[j] = i


def main():
    root     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    in_dir   = os.path.join(root, "instances", "overlap")
    out_dir  = os.path.join(root, "instances", "overlap_disjoint")
    differences = []

    for scenario in sorted(os.listdir(in_dir)):
        scenario_dir = os.path.join(in_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for p_folder in sorted(os.listdir(scenario_dir)):
            p_dir = os.path.join(scenario_dir, p_folder)
            if not os.path.isdir(p_dir):
                continue
            for corr_type in sorted(os.listdir(p_dir)):
                corr_dir = os.path.join(p_dir, corr_type)
                if not os.path.isdir(corr_dir):
                    continue

                out_subdir = os.path.join(out_dir, scenario, p_folder, corr_type)
                os.makedirs(out_subdir, exist_ok=True)

                for fname in sorted(os.listdir(corr_dir)):
                    if not fname.endswith(".txt"):
                        continue

                    in_path  = os.path.join(corr_dir, fname)
                    out_path = os.path.join(out_subdir, fname)

                    n, l, b, k, profits, weights, forfeit_sets     = read_instance(in_path)
                    n, l_new, b, k, profits, weights, forfeit_sets = make_disjoint(n, l, b, k, profits, weights, forfeit_sets)
                    differences.append(l_new-l)
                    write_instance(out_path, n, b, weights, profits, forfeit_sets, k)
                    print(f"Written: {out_path}")
    total = 0
    for dif in differences:
        total += dif
    print(f"Average number of new sets: {total / len(differences)}")

if __name__ == "__main__":
    main()
