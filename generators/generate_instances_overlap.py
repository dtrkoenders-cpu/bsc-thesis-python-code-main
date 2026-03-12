import os
import hashlib
import numpy as np

from generate_instances_kpdfs import generate_instance as _generate_disjoint, write_instance  # type: ignore


def add_overlap(forfeit_sets, n, p_overlap, rng):
    """
    Take a disjoint forfeit set partition and let each item independently
    join a second set with probability p_overlap.
    Returns an updated list of (items, h, d) tuples.
    """
    l = len(forfeit_sets)

    primary = np.full(n, -1, dtype=int)
    for i, (items, h, d) in enumerate(forfeit_sets):
        primary[items] = i

    secondary = np.full(n, -1, dtype=int)
    for j in range(n):
        if rng.random() < p_overlap:
            other = np.delete(np.arange(l), primary[j])
            secondary[j] = int(rng.choice(other))

    result = []
    for i, (items, h, d) in enumerate(forfeit_sets):
        extra     = np.where(secondary == i)[0]
        new_items = np.concatenate([items, extra]) if len(extra) > 0 else items
        result.append((new_items, h, d))

    return result


def generate_instance(n, corr_type, instance_id, scenario, p_overlap=0.3):
    b, weights, profits, forfeit_sets, k = _generate_disjoint(n, corr_type, instance_id, scenario)

    key  = f"overlap_{n}_{corr_type}_{instance_id}_{scenario}_{p_overlap}"
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
    rng  = np.random.default_rng(seed)

    forfeit_sets = add_overlap(forfeit_sets, n, p_overlap, rng)

    return b, weights, profits, forfeit_sets, k


def read_instance(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()

    n, l, b, k = map(int, lines[0].split())
    profits     = np.array(lines[1].split(), dtype=int)
    weights     = np.array(lines[2].split(), dtype=int)

    forfeit_sets = []
    idx = 3
    for _ in range(l):
        h, d, _ = map(int, lines[idx].split())
        items    = np.array(lines[idx + 1].split(), dtype=int)
        forfeit_sets.append((items, h, d))
        idx += 2

    return n, l, b, k, profits, weights, forfeit_sets


def main():
    scenarios   = [1, 2, 3, 4]
    n_values    = [300, 500, 700, 800, 1000]
    corr_types  = ["not-correlated", "correlated", "fully-correlated"]
    n_instances = 10
    p_overlaps  = [0.2, 0.4, 0.6]
    base_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "instances", "overlap")

    for scenario in scenarios:
        for p in p_overlaps:
            for corr_type in corr_types:
                os.makedirs(os.path.join(base_dir, f"scenario {scenario}", f"p_{int(p * 100)}", corr_type), exist_ok=True)

    for scenario in scenarios:
        for n in n_values:
            for p in p_overlaps:
                p_pct = int(p * 100)
                for corr_type in corr_types:
                    for idx in range(1, n_instances + 1):
                        b, weights, profits, forfeit_sets, k = generate_instance(n, corr_type, idx, scenario, p)
                        nS    = len(forfeit_sets)
                        fname = (
                            f"scen_{scenario}"
                            f"id_{idx}"
                            f"_objs_{n}"
                            f"_size_{b}"
                            f"_sets_{nS}"
                            f"_k_{k}"
                            f"_overlap_{p_pct}"
                            f"_{corr_type}.txt"
                        )
                        folder = os.path.join(base_dir, f"scenario {scenario}", f"p_{p_pct}", corr_type)
                        write_instance(os.path.join(folder, fname), n, b, weights, profits, forfeit_sets, k)
                        print(f"Written: {os.path.join(folder, fname)}")


if __name__ == "__main__":
    main()