import os
import hashlib
import numpy as np


def make_seed(n, corr_type, instance_idx):
    """Deterministic seed stable across Python runs (unlike built-in hash())."""
    key = f"{n}_{corr_type}_{instance_idx}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


def generate_forfeit_sets(rng, n, scenario):
    """
    Shuffle all item indices and partition into disjoint sets.
    Set sizes sampled uniformly from [2, max(2, n//50)].
    Returns list of np.arrays (item indices per set).
    """
    indices = np.arange(n)
    rng.shuffle(indices)

    if scenario in [1,3]:    
        max_size = max(2, n // 50)
    elif scenario in [2,4]:
        max_size = max(2, n // 20)
    else:
        raise ValueError(f"scenario must be 1, 2, 3 or 4, got {scenario}")

    sets = []
    pos = 0
    while pos < n:
        remaining = n - pos
        if remaining == 1:
            # Append the lone item to the last set to preserve the partition
            sets[-1] = np.append(sets[-1], indices[pos])
            break
        s = int(rng.integers(2, max_size + 1))
        s = min(s, remaining)
        sets.append(indices[pos : pos + s])
        pos += s
    return sets


def generate_instance(n, corr_type, instance_idx, scenario):
    seed = make_seed(n, corr_type, instance_idx)
    rng = np.random.default_rng(seed)

    w_min = 1
    w_max = 30
    # Knapsack capacity: (w_min + w_max) / 2 * n / 10
    b = int(((w_min + w_max) / 2) * (n / 10))

    # Item weights
    weights = rng.integers(w_min, w_max + 1, size=n)  # [1, 30]

    # Item profits
    if corr_type == "not-correlated":
        profits = rng.integers(w_min, w_max + 1, size=n)
    else:  # correlated or fully-correlated
        profits = weights + 10

    # Disjoint forfeit sets
    raw_sets = generate_forfeit_sets(rng, n, scenario)

    forfeit_sets = []  # list of (items, allowance h, forfeit cost d)
    for items in raw_sets:
        s = len(items)
        if scenario in [3,4]:
            h_max = max(1, int(np.floor((2 / 3) * s)))
            h = int(rng.integers(1, h_max + 1))
        elif scenario in [1,2]:
            h = 1
        else:
            raise ValueError(f"scenario must be 1, 2, 3 or 4, got {scenario}")

        if corr_type == "fully-correlated":
            # d_i = floor( sum of weights of (h_i + 1) highest-profit items / |C_i| )
            top_idx = np.argpartition(profits[items], -(h + 1))[-(h + 1):]
            d = int(np.floor(weights[items[top_idx]].sum() / s))
        else:
            d = int(rng.integers(1, 21))  # [1, 20]

        forfeit_sets.append((items, h, d))

    # Maximum violations k
    if scenario in [1,2]:
        k_map = {300: 15, 500: 25, 700: 35, 800: 45, 1000: 55}
        k = int(np.round(n / k_map[n], 0))
    elif scenario in [3,4]: 
        k = int(np.round(n/15, 0))
    else:
        raise ValueError(f"scenario must be 1, 2, 3 or 4, got {scenario}")


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
    scenarios = [1,2,3,4]
    n_values = [300, 500, 700, 800, 1000]
    corr_types = ["not-correlated", "correlated", "fully-correlated"]
    n_instances = 10
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instances", "disjoint")

    for scenario in scenarios:
        for corr_type in corr_types:
            os.makedirs(os.path.join(base_dir, "scenario " + str(scenario), corr_type), exist_ok=True)

    for scenario in scenarios:
        for n in n_values:
            for corr_type in corr_types:
                for idx in range(1, n_instances + 1):
                    b, weights, profits, forfeit_sets, k = generate_instance(n, corr_type, idx, scenario)
                    nS = len(forfeit_sets)
                    filename = (
                        f"scen_{scenario}"
                        f"id_{idx}"
                        f"_objs_{n}"
                        f"_size_{b}"
                        f"_sets_{nS}"
                        f"_k_{k}"
                        f"_{corr_type}.txt"
                    )
                    filepath = os.path.join(base_dir ,"scenario " + str(scenario), corr_type, filename)

                    write_instance(filepath, n, b, weights, profits, forfeit_sets, k)
                    print(f"Written: {filepath}")


if __name__ == "__main__":
    main()