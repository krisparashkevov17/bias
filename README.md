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

Scripts `01`–`03` are independent and can run in any order. `04_figures.py`
must run after them — it reads their CSV outputs.

| Script | Produces |
|---|---|
| `01_topology.py` | `data/topology_metrics.csv`, `data/topology_comparisons.csv` (SQ1) |
| `02_bias_analysis.py` | `data/stats/wilcoxon_gender_{F,M}.csv` |
| `03_sq2_analysis.py` | `data/topology_bias_data.csv`, `data/robustness/robustness_weighted_path_lme.csv`, `data/stats/lme_pooled_{main,interaction}.txt` (SQ2) |
| `04_figures.py` | all thesis figures (PDF) in `figures/` |
| `make_results_summary.py` | `results_summary.txt` — all reported numbers in one file (optional) |

Full reproduction:

```
python3 01_topology.py
python3 02_bias_analysis.py
python3 03_sq2_analysis.py
python3 04_figures.py
```
