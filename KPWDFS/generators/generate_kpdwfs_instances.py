"""
Instance generator for the Knapsack Problem with Disjoint Weight-based Forfeit Sets
(KPDWFS).

Weight-based interpretation of h and k
---------------------------------------
Unlike KPDFS (where h_i is an item-count allowance and k bounds the total number
of excess items), in KPDWFS every allowance is measured in weight units:

  - h_i  : maximum total weight of selected items allowed in forfeit set C_i
            before a penalty is charged.
  - k    : upper bound on the sum of per-set excess weights
            sum_i max(0, sum_{j in C_i} w_j * x_j  -  h_i)  <=  k

h_i is derived by converting the item-count allowance used in KPDFS to a weight
allowance by multiplying the count by the mean item weight in the set:
    h_i = floor(W_tot_i / |C_i|) * h_count
where h_count is 1 for Scenarios 1-2, and drawn from [1, floor(2/3*|C_i|)] for
Scenarios 3-4 (same distribution as KPDFS).

k is obtained by scaling the KPDFS item-count k by round(w_avg) = 16
(w_avg = (1+30)/2 = 15.5, rounded to 16).

All other aspects (item generation, forfeit-set structure, correlation types,
scenario parameters, file format) are identical to the KPDFS generator.
"""

import os
import hashlib
import numpy as np


W_AVG_ROUNDED = round((1 + 30) / 2)   # = 16


def make_seed(n, corr_type, instance_idx):
    """Deterministic seed stable across Python runs (unlike built-in hash())."""
    key = f"{n}_{corr_type}_{instance_idx}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


def generate_forfeit_sets(rng, n, scenario):
    """
    Shuffle all item indices and partition into disjoint sets.
    Set sizes sampled uniformly from [2, max(2, n//50)] (scenarios 1 & 3)
    or [2, max(2, n//20)] (scenarios 2 & 4).
    Returns list of np.arrays (item indices per set).
    """
    indices = np.arange(n)
    rng.shuffle(indices)

    if scenario in [1, 3]:
        max_size = max(2, n // 50)
    elif scenario in [2, 4]:
        max_size = max(2, n // 20)
    else:
        raise ValueError(f"scenario must be 1, 2, 3 or 4, got {scenario}")

    sets = []
    pos = 0
    while pos < n:
        remaining = n - pos
        if remaining == 1:
            sets[-1] = np.append(sets[-1], indices[pos])
            break
        s = int(rng.integers(2, max_size + 1))
        s = min(s, remaining)
        sets.append(indices[pos: pos + s])
        pos += s
    return sets


def generate_instance(n, corr_type, instance_idx, scenario):
    """
    Generate one KPDWFS instance.

    Returns
    -------
    b             : knapsack capacity
    weights       : np.array of item weights
    profits       : np.array of item profits
    forfeit_sets  : list of (items, h_i, d_i) where h_i is a WEIGHT allowance
    k             : upper bound on total excess weight
    """
    seed = make_seed(n, corr_type, instance_idx)
    rng = np.random.default_rng(seed)

    w_min, w_max = 1, 30
    b = int(((w_min + w_max) / 2) * (n / 10))

    weights = rng.integers(w_min, w_max + 1, size=n)

    if corr_type == "not-correlated":
        profits = rng.integers(w_min, w_max + 1, size=n)
    else:
        profits = weights + 10

    raw_sets = generate_forfeit_sets(rng, n, scenario)

    forfeit_sets = []
    for items in raw_sets:
        s = len(items)
        W_tot = int(weights[items].sum())
        mean_w = int(np.floor(W_tot / s))   # floor(average weight in set)

        # Item-count allowance (h_count) — same distribution as KPDFS
        if scenario in [1, 2]:
            h_count = 1
        else:  # scenarios 3 & 4
            h_max_count = max(1, int(np.floor((2 / 3) * s)))
            h_count = int(rng.integers(1, h_max_count + 1))

        # Weight allowance: scale count allowance by mean item weight in set
        h = mean_w * h_count

        assert h <= W_tot, (
            f"h={h} exceeds total set weight W_tot={W_tot} "
            f"(s={s}, mean_w={mean_w}, h_count={h_count})"
        )

        if corr_type == "fully-correlated":
            # Use h_count+1 highest-profit items (count-based, same as KPDFS)
            top_count = min(h_count + 1, s)
            top_idx = np.argpartition(profits[items], -top_count)[-top_count:]
            d = int(np.floor(weights[items[top_idx]].sum() / s))
        else:
            d = int(rng.integers(1, 21))  # uniform in [1, 20]

        forfeit_sets.append((items, h, d))

    # Global bound k: scale KPDFS item-count k by W_AVG_ROUNDED (=16)
    if scenario in [1, 2]:
        k_map = {300: 15, 500: 25, 700: 35, 800: 45, 1000: 55}
        k_kpdfs = int(np.round(n / k_map[n], 0))
    else:  # scenarios 3 & 4
        k_kpdfs = int(np.round(n / 15, 0))

    k = k_kpdfs * W_AVG_ROUNDED

    return b, weights, profits, forfeit_sets, k


def write_instance(filepath, n, b, weights, profits, forfeit_sets, k):
    nS = len(forfeit_sets)
    with open(filepath, "w") as f:
        f.write(f"{n} {nS} {b} {k}\n")
        f.write(" ".join(map(str, profits)) + "\n")
        f.write(" ".join(map(str, weights)) + "\n")
        for items, h, d in forfeit_sets:
            f.write(f"{h} {d} {len(items)}\n")
            f.write(" ".join(map(str, items)) + "\n")


def main():
    scenarios = [1, 2, 3, 4]
    n_values = [300, 500, 700, 800, 1000]
    corr_types = ["not-correlated", "correlated", "fully-correlated"]
    n_instances = 10
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "instances", "disjoint"
    )

    for scenario in scenarios:
        for corr_type in corr_types:
            os.makedirs(
                os.path.join(base_dir, "scenario " + str(scenario), corr_type),
                exist_ok=True,
            )

    for scenario in scenarios:
        for n in n_values:
            for corr_type in corr_types:
                for idx in range(1, n_instances + 1):
                    b, weights, profits, forfeit_sets, k = generate_instance(
                        n, corr_type, idx, scenario
                    )
                    nS = len(forfeit_sets)
                    filename = (
                        f"kpdwfs_sc{scenario}"
                        f"_id{idx}"
                        f"_n{n}"
                        f"_b{b}"
                        f"_sets{nS}"
                        f"_k{k}"
                        f"_{corr_type}.txt"
                    )
                    filepath = os.path.join(
                        base_dir, "scenario " + str(scenario), corr_type, filename
                    )
                    write_instance(filepath, n, b, weights, profits, forfeit_sets, k)
                    print(f"Written: {filepath}")


if __name__ == "__main__":
    main()
