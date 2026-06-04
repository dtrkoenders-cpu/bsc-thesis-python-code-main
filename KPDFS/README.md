# Knapsack Problem with Forfeit Sets — Instance Generator & Solvers

## Problem description

The **Knapsack Problem with Forfeit Sets (KPFS)** extends the classical 0/1 knapsack problem. Items belong to *forfeit sets*. Each set has an *allowance* `h_i`: if more than `h_i` items from set `i` are selected, a *forfeit cost* `d_i` is incurred per extra item. The objective is to maximise total profit minus total forfeit costs, subject to the knapsack weight capacity `b` and a global limit on total violations `k`.

This project studies two variants:
- **Disjoint (KPDFS)** — each item belongs to exactly one forfeit set; solvable exactly via DP.
- **Overlapping (KPFS)** — each item may belong to up to two forfeit sets; solved via a disjointification heuristic.

---

## Project structure

```
generators/
  generate_instances_kpdfs.py     # generate disjoint instances
  generate_instances_overlap.py   # generate overlapping instances

transforms/
  make_disjoint.py                # transform overlapping → disjoint instances

test/
  test_compare_overlap.py         # quick comparison: DP heuristic vs Gurobi on a subset

Solvers/
  solve_kpdfs_instance_dp.py      # DP solver for a single disjoint instance
  solve_kpfs_instance_gurobi.py   # Gurobi MIP solver for a single instance
  solve_all_instances_dp.py       # batch DP solver (disjoint instances)
  solve_all_instances_gurobi.py   # batch Gurobi solver (overlap instances)
  solve_all_instances_dp_overlap.py  # batch overlap solver (DP + overlap evaluation)

instances/
  disjoint/                       # output of generate_instances_kpdfs.py
  overlap/                        # output of generate_instances_overlap.py
  overlap_disjoint/               # output of make_disjoint.py

results/                          # CSV output files
```

---

## Requirements

- Python 3.8+
- NumPy
- Gurobi with valid licence (only for `solve_kpfs_instance_gurobi.py` and `solve_all_instances_gurobi.py`)

```
pip install numpy gurobipy
```

---

## Instance file format

All instance files share the same format:

```
nI nS b k
p_0 p_1 ... p_{nI-1}
w_0 w_1 ... w_{nI-1}
h_0 d_0 |C_0|
j_0 j_1 ... j_{|C_0|-1}
h_1 d_1 |C_1|
...
```

| Symbol    | Meaning                          |
|-----------|----------------------------------|
| `nI`      | Number of items                  |
| `nS`      | Number of forfeit sets           |
| `b`       | Knapsack capacity                |
| `k`       | Maximum total violations allowed |
| `p_j`     | Profit of item `j`               |
| `w_j`     | Weight of item `j`               |
| `h_i`     | Allowance of forfeit set `i`     |
| `d_i`     | Forfeit cost of set `i`          |
| `\|C_i\|` | Cardinality of forfeit set `i`   |

---

## Disjoint instances

### Generation

```bash
python generators/generate_instances_kpdfs.py
```

Generates 4 scenarios × 3 correlation types × 5 item counts × 10 instances under `instances/disjoint/`.

```
instances/disjoint/
  scenario 1/
    not-correlated/
    correlated/
    fully-correlated/
  scenario 2/  ...
```

### Generation parameters

| Parameter       | Value                            |
|-----------------|----------------------------------|
| Item counts `n` | 300, 500, 700, 800, 1000         |
| Capacity `b`    | `floor(15.5 * n / 10)`           |
| Weight range    | Uniform [1, 30]                  |
| Instances       | 10 per case                      |

### Scenarios

| Scenario | Set size range          | Allowance `h_i`                   | `k`                   |
|----------|-------------------------|-----------------------------------|-----------------------|
| 1        | [2, max(2, n // 50)]    | Fixed at 1                        | `round(n / k_map[n])` |
| 2        | [2, max(2, n // 20)]    | Fixed at 1                        | `round(n / k_map[n])` |
| 3        | [2, max(2, n // 50)]    | Uniform [1, floor(2/3 · \|C_i\|)] | `round(n / 15)`       |
| 4        | [2, max(2, n // 20)]    | Uniform [1, floor(2/3 · \|C_i\|)] | `round(n / 15)`       |

`k_map = {300: 15, 500: 25, 700: 35, 800: 45, 1000: 55}`

### Correlation types

- **not-correlated** — profits and weights drawn independently from Uniform[1, 30]; forfeit costs from Uniform[1, 20].
- **correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; forfeit costs from Uniform[1, 20].
- **fully-correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; `d_i = floor(sum of weights of the h_i + 1 highest-profit items in C_i / |C_i|)`.

---

## Overlapping instances

### Generation

```bash
python generators/generate_instances_overlap.py
```

Generates instances where each item independently joins a second forfeit set with probability `p_overlap`. Scenarios and correlation types match the disjoint generator. Overlap probabilities: 0.2, 0.4, 0.6.

```
instances/overlap/
  scenario 1/
    p_20/
      not-correlated/
      correlated/
      fully-correlated/
    p_40/  ...
    p_60/  ...
  scenario 2/  ...
```

### Disjointification

```bash
python transforms/make_disjoint.py
```

Transforms every overlapping instance into an equivalent disjoint one and writes results to `instances/overlap_disjoint/` (mirroring the `overlap/` structure).

For each pair of sets `(C_a, C_b)` sharing items `S_ab`:
- A new merged set `C_ab` is created with `items = S_ab`, `d = d_a + d_b`, and `h` determined by scenario (see table below).
- `C_a` and `C_b` keep only their non-shared items; allowances are scaled proportionally: `h' = floor(h * |remaining| / |original|)`.
- Empty sets after removal are dropped.
- The global violation budget `k` is scaled by `K_SCALE` after transformation.

Both `K_SCALE` and `MERGED_H` are configurable constants at the top of `make_disjoint.py`.

| Scenario | Merged set allowance `h`  | `K_SCALE` |
|----------|---------------------------|-----------|
| 1        | `\|S_ab\| − 1`            | 1         |
| 2        | `\|S_ab\| − 1`            | 1         |
| 3        | `min(h_a, h_b)`           | 1/4       |
| 4        | `min(h_a, h_b)`           | 1/6       |

---

## DP solver (disjoint instances)

### Solving a single instance

```bash
python Solvers/solve_kpdfs_instance_dp.py instances/disjoint/scenario\ 1/not-correlated/...txt
```

Prints per-set progress and a final objective value.

### Algorithm

The DP maintains a table `f[b', k']` = best profit using total weight `b'` and at most `k'` violations across all sets processed so far.

For each forfeit set `C_i`:
1. Compute `A_i[W, s]` = max profit from exactly `s` items with total weight exactly `W`.
2. Compute `val_i[W, s] = A_i[W, s] − d_i · max(0, s − h_i)`.
3. Update `f_curr` via the Bellman equation over all finite `(W, s)` states.

Final answer: `max over b' of f[b', k]`.

Free items (not in any forfeit set) are collected into a penalty-free extra set.

### Backtracking

Pass `return_items=True` to also recover the selected item indices:

```python
from solve_kpdfs_instance_dp import solve_kpdfs

result = solve_kpdfs("path/to/instance.txt", return_items=True)
print(result["obj_value"])
print(result["selected_items"])   # list of item indices
```

### Batch (disjoint)

```bash
python Solvers/solve_all_instances_dp.py
```

Writes `results/results_dp.csv` with columns: `scenario`, `corr_type`, `instance_file`, `n`, `obj_value`, `runtime`.

---

## Gurobi MIP solver

Solves a single KPFS instance (disjoint or overlapping) as a binary integer program.

```bash
python Solvers/solve_kpfs_instance_gurobi.py path/to/instance.txt
```

### Model formulation

- Binary variables `x_j ∈ {0, 1}` — 1 if item `j` is selected.
- Continuous variables `v_i ≥ 0` — violations for forfeit set `i`.
- Objective: maximise `Σ p_j x_j − Σ d_i v_i`.
- Constraints: weight budget, global violation bound `k`, per-set forfeit constraint.
- Upper bounds on `v_i` tightened via Proposition 2.2.

| Parameter  | Value            |
|------------|------------------|
| Time limit | 3 hours (10800s) |
| MIP gap    | 1e-4             |

### Batch (overlap, Gurobi)

```bash
python Solvers/solve_all_instances_gurobi.py
```

Walks `instances/overlap/` and writes `results/results_overlap_gurobi.csv` with columns: `scenario`, `p_overlap`, `corr_type`, `instance_file`, `n`, `obj_value`, `total_violations`, `runtime`.

---

## Overlap evaluation (DP heuristic)

```bash
python Solvers/solve_all_instances_dp_overlap.py
```

For each instance in `instances/overlap_disjoint/`:
1. Solves the disjointified instance with DP and recovers the selected items.
2. Evaluates those items against the **original overlapping** forfeit sets: computes `Σ p_j − Σ d_i · max(0, |selected ∩ C_i| − h_i)`, returning `None` if total violations exceed `k`.

Writes `results/results_overlap_dp.csv` with columns:

| Column          | Content                                               |
|-----------------|-------------------------------------------------------|
| `scenario`      | Scenario number                                       |
| `p_overlap`     | Overlap probability folder (e.g. `p_20`)              |
| `corr_type`     | Correlation type                                      |
| `instance_file` | Filename                                              |
| `n`             | Number of items                                       |
| `obj_disjoint`  | DP objective on disjointified instance                |
| `obj_overlap`   | Objective of the same solution on the original KPFS   |
| `runtime`       | DP wall-clock seconds                                 |

---

## Reproducibility

Each instance is generated from a deterministic seed derived via MD5 from `(n, corr_type, instance_idx)`, ensuring identical instances across runs regardless of `PYTHONHASHSEED`.
