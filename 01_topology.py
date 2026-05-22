"""
01_topology.py — Compute ASPL and CC for each semantic network (thesis SQ1).

ASPL — unweighted, sampled

CC — unweighted

Statistical comparisons
  - Non-parametric bootstrap (1,000 resamples, 95 % CI) 

Output
------
  data/topology_metrics.csv      — ASPL and CC per network, with bootstrap CIs
  data/topology_comparisons.csv  — pairwise Mann-Whitney U results vs. Humans

Run:
  python3 01_topology.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    DATA_DIR, MODELS,
    load_edgelist, bootstrap_metric,
)

N_SAMPLES   = 1000   # source nodes sampled for ASPL
N_BOOTSTRAP = 1000   # resamples for 95 % CI
SEED        = 42


# ── 1. Load networks 
print("Loading filtered networks …")
graphs = {}
for model in MODELS:
    g = load_edgelist(model, directed=False)
    graphs[model] = g
    print(f"  {model}: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")


# ── 2. ASPL (unweighted, igraph BFS, sampled) 
print(f"\nEstimating ASPL ({N_SAMPLES:,} source nodes per network) …", flush=True)

aspl_src_means = {}

for model, g in graphs.items():
    ig_g = ig.Graph.from_networkx(g)
    n    = ig_g.vcount()
    rng  = random.Random(SEED)
    idxs = rng.sample(range(n), min(N_SAMPLES, n))

    src_means = []
    for src in idxs:
        # distances() returns a list; filter out self (d=0) and disconnected (inf).
        dists  = ig_g.distances(source=src, mode="all")[0]
        finite = [d for d in dists if 0 < d < float("inf")]
        if finite:
            src_means.append(np.mean(finite))

    aspl_src_means[model] = src_means
    print(f"  {model}: ASPL ≈ {np.mean(src_means):.4f}  ({len(src_means):,} valid sources)",
          flush=True)


# ── 3. Clustering coefficient (unweighted, Watts-Strogatz) ───────────────────
# Per-node values are stored so that bootstrapping and Mann-Whitney tests
# operate on the full distribution rather than a single summary statistic.
print("\nComputing clustering coefficients …", flush=True)

cc_uw_values = {}   # unweighted (Watts-Strogatz)

for model, g in graphs.items():
    cc_uw_values[model] = list(nx.clustering(g).values())
    print(f"  {model}: CC_unweighted = {np.mean(cc_uw_values[model]):.6f}", flush=True)


# ── 4. Bootstrap 95 % CIs ────────────────────────────────────────────────────
print(f"\nBootstrapping CIs ({N_BOOTSTRAP:,} resamples) …")

boot_aspl = {}
boot_cc_uw = {}

for model in MODELS:
    boot_aspl[model]  = bootstrap_metric(aspl_src_means[model], N_BOOTSTRAP, SEED)
    boot_cc_uw[model] = bootstrap_metric(cc_uw_values[model],   N_BOOTSTRAP, SEED)

    m_a, lo_a, hi_a = boot_aspl[model]
    m_c, lo_c, hi_c = boot_cc_uw[model]
    print(f"  {model}: ASPL {m_a:.4f} [{lo_a:.4f}, {hi_a:.4f}]  "
          f"CC {m_c:.6f} [{lo_c:.6f}, {hi_c:.6f}]")


# ── 5. Pairwise comparisons vs. Humans (Mann-Whitney U) ──────────────────────
# Mann-Whitney U is used because per-node CC distributions are right-skewed
# and a t-test would be inappropriate.  Two-sided alternative tests whether
# the distributions differ in either direction.
print("\n─── Pairwise comparisons vs. Humans ─────────────────────────────────")

comparisons = []
for model in MODELS[1:]:
    u_aspl,  p_aspl  = stats.mannwhitneyu(aspl_src_means["Humans"], aspl_src_means[model],
                                          alternative="two-sided")
    u_cc_uw, p_cc_uw = stats.mannwhitneyu(cc_uw_values["Humans"],   cc_uw_values[model],
                                          alternative="two-sided")

    delta_aspl  = np.mean(aspl_src_means["Humans"]) - np.mean(aspl_src_means[model])
    delta_cc_uw = np.mean(cc_uw_values["Humans"])   - np.mean(cc_uw_values[model])

    comparisons.append({
        "comparison":  f"Humans vs {model}",
        "ASPL_human":  boot_aspl["Humans"][0],
        "ASPL_llm":    boot_aspl[model][0],
        "ASPL_delta":  delta_aspl,
        "ASPL_U":      u_aspl,
        "ASPL_p":      p_aspl,
        "CC_uw_human": boot_cc_uw["Humans"][0],
        "CC_uw_llm":   boot_cc_uw[model][0],
        "CC_uw_delta": delta_cc_uw,
        "CC_uw_U":     u_cc_uw,
        "CC_uw_p":     p_cc_uw,
    })
    print(f"  Humans vs {model}: "
          f"ΔASPL={delta_aspl:+.4f} (p={p_aspl:.4f})  "
          f"ΔCC_uw={delta_cc_uw:+.6f} (p={p_cc_uw:.4f})")


# ── 6. Save ───────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    "model":         MODELS,
    "ASPL":          [boot_aspl[m][0]  for m in MODELS],
    "ASPL_CI_lo":    [boot_aspl[m][1]  for m in MODELS],
    "ASPL_CI_hi":    [boot_aspl[m][2]  for m in MODELS],
    "CC_unweighted": [boot_cc_uw[m][0] for m in MODELS],
    "CC_uw_CI_lo":   [boot_cc_uw[m][1] for m in MODELS],
    "CC_uw_CI_hi":   [boot_cc_uw[m][2] for m in MODELS],
})

metrics_out = os.path.join(DATA_DIR, "topology_metrics.csv")
comp_out    = os.path.join(DATA_DIR, "topology_comparisons.csv")
metrics_df.to_csv(metrics_out, index=False)
pd.DataFrame(comparisons).to_csv(comp_out, index=False)

print(f"\n─── Summary ──────────────────────────────────────────────────────────")
print(metrics_df.to_string(index=False))
print(f"\nSaved: {metrics_out}")
print(f"Saved: {comp_out}")
