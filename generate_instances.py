import os
import hashlib
import numpy as np


def make_seed(n, corr_type, instance_idx):
    """Deterministic seed stable across Python runs (unlike built-in hash())."""
    key = f"{n}_{corr_type}_{instance_idx}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)


def generate_forfeit_sets(rng, n):
    """
    Shuffle all item indices and greedily partition into disjoint sets.
    Set sizes sampled uniformly from [2, max(2, n//50)].
    Returns list of np.arrays (item indices per set).
    """
    indices = np.arange(n)
    rng.shuffle(indices)

    max_size = max(2, n // 50)
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


def generate_instance(n, corr_type, instance_idx):
    seed = make_seed(n, corr_type, instance_idx)
    rng = np.random.default_rng(seed)

    # Knapsack capacity: (w_min + w_max) / 2 * n / 10
    b = int(((1 + 30) / 2) * (n / 10))

    # Item weights
    weights = rng.integers(1, 31, size=n)  # [1, 30]

    # Item profits
    if corr_type == "not-correlated":
        profits = rng.integers(1, 31, size=n)
    else:  # correlated or fully-correlated
        profits = weights + 10

    # Disjoint forfeit sets
    raw_sets = generate_forfeit_sets(rng, n)

    forfeit_sets = []  # list of (items, allowance h, forfeit cost d)
    for items in raw_sets:
        s = len(items)
        h_max = max(1, int(np.floor((2 / 3) * s)))
        h = int(rng.integers(1, h_max + 1))

        if corr_type == "fully-correlated":
            # d_i = floor( sum of weights of (h_i + 1) highest-profit items / |C_i| )
            top_idx = np.argpartition(profits[items], -(h + 1))[-(h + 1):]
            d = int(np.floor(weights[items[top_idx]].sum() / s))
        else:
            d = int(rng.integers(1, 21))  # [1, 20]

        forfeit_sets.append((items, h, d))

    return b, weights, profits, forfeit_sets


def write_instance(filepath, n, b, weights, profits, forfeit_sets):
    nS = len(forfeit_sets)
    with open(filepath, "w") as f:
        f.write(f"{n} {nS} {b}\n")
        f.write(" ".join(map(str, profits)) + "\n")
        f.write(" ".join(map(str, weights)) + "\n")
        for items, h, d in forfeit_sets:
            f.write(f"{h} {d} {len(items)}\n")
            f.write(" ".join(map(str, items)) + "\n")


def main():
    n_values = [300, 500, 700, 800, 1000]
    corr_types = ["not-correlated", "correlated", "fully-correlated"]
    n_instances = 10
    base_dir = "instances"

    for corr_type in corr_types:
        os.makedirs(os.path.join(base_dir, corr_type), exist_ok=True)

    for n in n_values:
        max_num_conflicts = max(2, n // 50)
        for corr_type in corr_types:
            for idx in range(1, n_instances + 1):
                b, weights, profits, forfeit_sets = generate_instance(n, corr_type, idx)

                nS = len(forfeit_sets)
                max_cost = max(d for _, _, d in forfeit_sets)
                filename = (
                    f"id_{idx}"
                    f"_objs_{n}"
                    f"_size_{b}"
                    f"_sets_{nS}"
                    f"_maxNumConflicts_{max_num_conflicts}"
                    f"_maxCost_{max_cost}"
                    f"_{corr_type}.txt"
                )
                filepath = os.path.join(base_dir, corr_type, filename)

                write_instance(filepath, n, b, weights, profits, forfeit_sets)
                print(f"Written: {filepath}")


if __name__ == "__main__":
    main()
