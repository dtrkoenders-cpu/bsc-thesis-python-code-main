# Knapsack Problem with Forfeit Sets — Instance Generator & Solver

## Problem description

The **Knapsack Problem with Forfeit Sets (KPFS)** extends the classical 0/1 knapsack problem. Items are partitioned into disjoint *forfeit sets*. Each set has an *allowance* `h_i`: if more than `h_i` items from set `i` are selected, a *forfeit cost* `d_i` is incurred per violation. The objective is to maximise total profit minus total forfeit costs, subject to the knapsack weight capacity and a global limit on total violations.

## Requirements

- Python 3.8+
- NumPy
- Gurobi (with valid licence)

```
pip install numpy gurobipy
```

## Scripts

| Script | Purpose |
|---|---|
| `generate_instances.py` | Generate all benchmark instances |
| `solve_kpfs_instance_gurobi.py` | Solve a single instance with Gurobi |
| `solve_all_instances.py` | Solve all instances and write results to CSV |

---

## Generating instances

```bash
python generate_instances.py
```

Generates 150 instance files (5 item counts × 3 correlation types × 10 instances) under the `instances/` directory.

### Output structure

```
instances/
  not-correlated/
  correlated/
  fully-correlated/
```

Each file is named:

```
id_{i}_objs_{n}_size_{b}_sets_{nS}_maxNumConflicts_{c}_maxCost_{d}_{corr_type}.txt
```

| Field             | Meaning                                        |
|-------------------|------------------------------------------------|
| `id`              | Instance index (1–10)                          |
| `objs`            | Number of items `n`                            |
| `size`            | Knapsack capacity `b`                          |
| `sets`            | Number of forfeit sets                         |
| `maxNumConflicts` | Upper bound on set size used during generation |
| `maxCost`         | Maximum forfeit cost `d_i` across all sets     |
| `corr_type`       | Correlation type (see below)                   |

### Instance file format

```
nI nS kS
p_0 p_1 ... p_{nI-1}
w_0 w_1 ... w_{nI-1}
h_0 d_0 |C_0|
j_0 j_1 ... j_{|C_0|-1}
h_1 d_1 |C_1|
...
```

| Symbol  | Meaning                        |
|---------|--------------------------------|
| `nI`    | Number of items                |
| `nS`    | Number of forfeit sets         |
| `kS`    | Knapsack capacity              |
| `p_j`   | Profit of item `j`             |
| `w_j`   | Weight of item `j`             |
| `h_i`   | Allowance of forfeit set `i`   |
| `d_i`   | Forfeit cost of set `i`        |
| `\|C_i\|` | Cardinality of forfeit set `i` |

### Generation parameters

| Parameter          | Value                              |
|--------------------|------------------------------------|
| Item counts `n`    | 300, 500, 700, 800, 1000           |
| Capacity `b`       | `floor((1 + 30) / 2 * n / 10)`    |
| Instances per case | 10                                 |
| Weight range       | Uniform [1, 30]                    |
| Set size range     | Uniform [2, max(2, n // 50)]       |
| Allowance `h_i`    | Uniform [1, floor(2/3 * \|C_i\|)] |

### Correlation types

**not-correlated** — weights and profits drawn independently from Uniform[1, 30]; forfeit costs from Uniform[1, 20].

**correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; forfeit costs from Uniform[1, 20].

**fully-correlated** — weights from Uniform[1, 30]; profits `p_j = w_j + 10`; forfeit cost `d_i = floor(sum of weights of the h_i + 1 highest-profit items in C_i / |C_i|)`.

### Forfeit set construction

All `n` item indices are shuffled and greedily partitioned into disjoint sets, forming a true partition of all items. Set sizes are sampled from [2, max(2, n // 50)]; if a single item would be left over it is appended to the last set.

### Reproducibility

Each instance is generated from a deterministic seed derived from `(n, corr_type, instance_idx)` via MD5, ensuring identical instances across runs and machines regardless of Python's `PYTHONHASHSEED`.

---

## Solving a single instance

```bash
python solve_kpfs_instance_gurobi.py instances/not-correlated/id_1_objs_300_size_465_sets_76_maxNumConflicts_6_maxCost_20_not-correlated.txt
```

Add `--verbose` to enable Gurobi's solver output:

```bash
python solve_kpfs_instance_gurobi.py path/to/instance.txt --verbose
```

A live elapsed-time counter is shown during solving. After completion the script prints:

```
Status:           optimal
Runtime:          3.21s
Objective value:  4823.0000
MIP gap:          0.00e+00
Items selected:   47
Total violations: 2.0000
```

### Gurobi settings

| Parameter   | Value  |
|-------------|--------|
| Time limit  | 3 hours (10800s) |
| MIP gap     | 1e-4   |

### Model formulation

- Binary variables `x_j ∈ {0,1}` — 1 if item `j` is selected
- Continuous variables `v_i ≥ 0` — violations for forfeit set `i`
- Objective: maximise `Σ p_j x_j − Σ d_i v_i`
- Constraints: weight budget, global violation bound `k = round(n/15)`, per-set violation definition
- Upper bounds on `v_i` tightened via Proposition 2.2

### Importing as a module

```python
from solve_kpfs_instance_gurobi import solve_kpfs

result = solve_kpfs("path/to/instance.txt")
print(result["obj_value"])
print(result["selected_items"])
```

Returned dict keys: `status`, `obj_value`, `mip_gap`, `runtime`, `selected_items`, `violations_per_set`, `total_violations`.

---

## Solving all instances

```bash
python solve_all_instances.py
```

Iterates over all instance files in `instances/`, solves each with Gurobi, prints a one-line summary per instance, and writes all results to `results.csv`.

### results.csv columns

| Column            | Content                                      |
|-------------------|----------------------------------------------|
| `corr_type`       | not-correlated / correlated / fully-correlated |
| `instance_file`   | Filename                                     |
| `status`          | optimal / time_limit / infeasible            |
| `obj_value`       | Objective value                              |
| `mip_gap`         | Relative MIP gap                             |
| `runtime`         | Wall-clock seconds                           |
| `n_items_selected`| Number of selected items                     |
| `total_violations`| Sum of `v_i` across all forfeit sets         |

The CSV is flushed after every row, so partial results are preserved if the run is interrupted.
