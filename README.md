# Semantic network topology and gender bias in human and LLM networks

Analysis code for a bachelor's thesis comparing the human SWOW network with
three LLM-generated semantic networks (Mistral-7B, Llama3.1-8B,
Claude-3.5-Haiku). It extends Abramski et al. (2025) with network topology
metrics (SQ1) and a mixed-effects analysis of gender bias on weighted path
distance (SQ2).

The input networks, activation matrices, and the `FA_Functions.py` helpers
come from the LWOW repository (Abramski et al., 2025):
https://github.com/LLMWorldOfWords/LWOW

## Setup

```
python3 -m pip install -r requirements.txt
```

Developed on Python 3.9.

## Inputs (already provided)

The pipeline reads two pre-built sets of files; no preprocessing is re-run:

- `data/graphs/edge_lists/FA_{model}_edgelist.csv` — filtered semantic networks
- `LWOW/reproducibility/data/LDT_analyses/FA_matrices/{model}_gender.csv` —
  the spreading-activation matrices released by Abramski et al. (2025),
  loaded directly. Spreading activation is not recomputed here.

## Pipeline

Scripts `01`–`03` are independent of each other and can run in any order.
`04` and `make_results_summary.py` depend on their outputs. `05`, `06` and
`make_weight_distribution_figure.py` are independent.

| Script | Produces |
|---|---|
| `01_topology.py` | `data/topology_metrics.csv`, `data/topology_comparisons.csv` (SQ1) |
| `02_bias_analysis.py` | `data/stats/wilcoxon_gender_{F,M}.csv` |
| `03_robustness.py` | `data/topology_bias_data.csv`, `data/robustness/robustness_weighted_path_lme.csv`, `data/stats/lme_pooled_{main,interaction}.txt` (SQ2) |
| `04_figures.py` | 9 result figures in `figures/` |
| `05_network_figures.py` | `fig_networks.pdf`, `fig_networks_ego.pdf`, `fig_mindset_streams.pdf` |
| `06_network_extras.py` | `fig_network_overview.pdf`, `fig_spreading_example.pdf` |
| `make_results_summary.py` | `results_summary.txt` — all reported numbers in one file |
| `make_weight_distribution_figure.py` | `figures/fig_edge_weight_distribution.pdf` |

Full reproduction:

```
python3 01_topology.py
python3 02_bias_analysis.py
python3 03_robustness.py
python3 04_figures.py
python3 05_network_figures.py
python3 06_network_extras.py
python3 make_results_summary.py
python3 make_weight_distribution_figure.py
```

## Layout

```
utils.py                  shared constants, paths, graph/matrix loaders
01..06_*.py               analysis and figure scripts
make_*.py                 results summary + weight-distribution figure
data/                     input edge lists + generated CSVs
figures/                  generated PDF figures
LWOW/                     Abramski et al. (2025) reproducibility repo (inputs)
SWOW-EN18/                human free-association dataset
spreadpy_backup/          archived Python/SpreadPy replication (not used)
```
