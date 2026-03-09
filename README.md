# Knapsack Problem with Forfeit Sets — Instance Generator & Solvers

## Problem description

The **Knapsack Problem with Forfeit Sets (KPFS)** extends the classical 0/1 knapsack problem. Items are partitioned into disjoint *forfeit sets*. Each set has an *allowance* `h_i`: if more than `h_i` items from set `i` are selected, a *forfeit cost* `d_i` is incurred per violation. The objective is to maximise total profit minus total forfeit costs, subject to the knapsack weight capacity and a global limit on total violations `k`.

## Requirements

- Python 3.8+
- NumPy
- pandas
- Gurobi (with valid licence) — only required for the MIP solver

```
pip install numpy pandas gurobipy
```

## Scripts

| Script | Purpose |
|---|---|
| `generate_instances_kpdfs.py` | Generate all benchmark instances |
| `solve_kpfs_instance_gurobi.py` | Solve a single instance with Gurobi (ILP) |
| `solve_kpdfs_instance_dp.py` | Solve a single instance with dynamic programming |
| `solve_all_instances_gurobi.py` | Solve all instances with Gurobi, write `results_gurobi.csv` |
| `solve_all_instances_dp.py` | Solve all instances with DP, write `results_dp.csv` |

---

## Generating instances

```bash
python generate_instances_kpdfs.py
```

Generates instances for 4 scenarios × 3 correlation types × 5 item counts × 10 instances under the `instances/` directory.

### Output structure

```
instances/
  scenario 1/
    not-correlated/
    correlated/
    fully-correlated/
  scenario 2/
  scenario 3/
  scenario 4/
```

Each file is named:

```
scen_{scenario}id_{i}_objs_{n}_size_{b}_sets_{nS}_maxNumConflicts_{c}_maxCost_{d}_{corr_type}.txt
```

| Field             | Meaning                                        |
|-------------------|------------------------------------------------|
| `scen`            | Scenario 1, 2, 3 or 4                          |
| `id`              | Instance index (1–10)                          |
| `objs`            | Number of items `n`                            |
| `size`            | Knapsack capacity `b`                          |
| `sets`            | Number of forfeit sets                         |
| `maxNumConflicts` | Upper bound on set size used during generation |
| `maxCost`         | Maximum forfeit cost `d_i` across all sets     |
| `corr_type`       | Correlation type (see below)                   |

### Instance file format

```
nI nS kS k
p_0 p_1 ... p_{nI-1}
w_0 w_1 ... w_{nI-1}
h_0 d_0 |C_0|
j_0 j_1 ... j_{|C_0|-1}
h_1 d_1 |C_1|
...
```

| Symbol    | Meaning                           |
|-----------|-----------------------------------|
| `nI`      | Number of items                   |
| `nS`      | Number of forfeit sets            |
| `kS`      | Knapsack capacity                 |
| `k`       | Maximum total violations allowed  |
| `p_j`     | Profit of item `j`                |
| `w_j`     | Weight of item `j`                |
| `h_i`     | Allowance of forfeit set `i`      |
| `d_i`     | Forfeit cost of set `i`           |
| `\|C_i\|` | Cardinality of forfeit set `i`    |

### Generation parameters

| Parameter          | Value                              |
|--------------------|------------------------------------|
| Item counts `n`    | 300, 500, 700, 800, 1000           |
| Capacity `b`       | `floor((1 + 30) / 2 * n / 10)`     |
| Instances per case | 10                                 |
| Weight range       | Uniform [1, 30]                    |
| Set size range     | Scenario-dependent (see below)     |
| Allowance `h_i`    | Scenario-dependent (see below)     |

### Scenarios

| Scenario | Allowance `h_i`                      | `k`                        | Set size range                 |
|----------|--------------------------------------|----------------------------|--------------------------------|
| 1        | Fixed at 1                           | `round(n / k_map[n])`      | Uniform [2, max(2, n // 50)]   |
| 2        | Fixed at 1                           | `round(n / k_map[n])`      | Uniform [2, max(2, n // 20)]   |
| 3        | Uniform [1, floor(2/3 · \|C_i\|)]    | `round(n / 15)`            | Uniform [2, max(2, n // 50)]   |
| 4        | Uniform [1, floor(2/3 · \|C_i\|)]    | `round(n / 15)`            | Uniform [2, max(2, n // 20)]   |

### Correlation types

**not-correlated** — weights and profits drawn independently from Uniform[1, 30]; forfeit costs from Uniform[1, 20].

**correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; forfeit costs from Uniform[1, 20].

**fully-correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; forfeit cost `d_i = floor(sum of weights of the h_i + 1 highest-profit items in C_i / |C_i|)`.

### Forfeit set construction

All `n` item indices are shuffled and greedily partitioned into disjoint sets, forming a true partition of all items. Set sizes are sampled from [2, max(2, n // 50)] or [2, max(2, n // 20)] depending on the scenario; if a single item would be left over it is appended to the last set.

### Reproducibility

Each instance is generated from a deterministic seed derived from `(n, corr_type, instance_idx)` via MD5, ensuring identical instances across runs and machines regardless of Python's `PYTHONHASHSEED`.

---

## Gurobi LP solver

### Solving a single instance

```bash
python solve_kpfs_instance_gurobi.py instances/scenario\ 1/not-correlated/id_1_objs_300_...txt
```

Output after completion:

```
Runtime:          3.21s
Objective value:  4823.0000
```

### Model formulation

- Continuous variables `x_j ∈ [0, 1]` — 1 if item `j` is selected
- Continuous variables `v_i ≥ 0` — violations for forfeit set `i`
- Objective: maximise `Σ p_j x_j − Σ d_i v_i`
- Constraints: weight budget, global violation bound `k`, per-set violation definition
- Upper bounds on `v_i` tightened via Proposition 2.2

### Gurobi settings

| Parameter  | Value            |
|------------|------------------|
| Output     | Suppressed       |
| Time limit | 3 hours (10800s) |
| MIP gap    | 1e-4             |

### Solving all instances

```bash
python solve_all_instances_gurobi.py
```

Writes results to `results_gurobi.csv`.

| Column          | Content            |
|-----------------|--------------------|
| `scenario`      | Scenario number    |
| `corr_type`     | Correlation type   |
| `instance_file` | Filename           |
| `n`             | Number of items    |
| `obj_value`     | Objective value    |
| `runtime`       | Wall-clock seconds |

### Importing as a module

```python
from solve_kpfs_instance_gurobi import solve_kpfs

result = solve_kpfs("path/to/instance.txt")
print(result["obj_value"])
```

Returned dict keys: `obj_value`, `runtime`.

---

## Dynamic programming solver

### Solving a single instance

```bash
python solve_kpdfs_instance_dp.py instances/scenario\ 1/not-correlated/id_1_objs_300_...txt
```

Prints progress per forfeit set and a final objective value.

### Algorithm overview

The DP processes forfeit sets one by one, maintaining a table `f[b', k']` = best profit using exactly `b'` total weight and at most `k'` violations from the sets processed so far.

For each forfeit set `C_i`:
1. Compute `A_i[W, s]` = max profit from exactly `s` items in `C_i` with total weight exactly `W` (small 0/1 knapsack DP)
2. Compute `val_i[W, s] = A_i[W, s] − d_i · max(0, s − h_i)`
3. Update `f_curr[b', k']` via the Bellman equation over all valid `(W, s)` pairs using numpy slice operations

Final answer: `max over b' = 0..b of f[b', k]`.

Items not belonging to any forfeit set are collected into a penalty-free extra set.

### Solving all instances

```bash
python solve_all_instances_dp.py
```

Writes results to `results_dp.csv`.

| Column          | Content            |
|-----------------|--------------------|
| `scenario`      | Scenario number    |
| `corr_type`     | Correlation type   |
| `instance_file` | Filename           |
| `n`             | Number of items    |
| `obj_value`     | Objective value    |
| `runtime`       | Wall-clock seconds |

---