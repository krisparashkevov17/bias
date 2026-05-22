"""
03_robustness.py — Robustness checks for SQ2.

Experiments
-----------
1. Topology–bias correlation, weighted paths (LME per network)

2. Pooled LME (SQ2)

Output
  data/robustness/robustness_weighted_path_lme.csv
  data/topology_bias_data.csv
  data/stats/lme_pooled_main.txt
  data/stats/lme_pooled_interaction.txt

"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    MODELS, GENDER_PAIRS, PRIMES_F, PRIMES_M, TARGETS_F, TARGETS_M,
    DATA_DIR, ROBUSTNESS_DIR,
    load_edgelist, load_activation_matrix, normalizeDF,
)

STATS_DIR = os.path.join(DATA_DIR, "stats")
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Weighted shortest paths (distance = 1 / weight)
# ═══════════════════════════════════════════════════════════════════════════════
# distance = 1/weight, then Dijkstra gives the minimum-cost prime→target path.
# bias_score = matched-pair activation asymmetry (DiffF / DiffM, Abramski et al.
# 2025): AL of the consistent prime minus its GENDER_PAIRS partner.
print("══ Experiment 1: Weighted shortest paths  [Abramski R matrices] ════════════")

topo_rows  = []
wlme_rows  = []

for model in MODELS:
    g       = load_edgelist(model, directed=False)
    df_norm = normalizeDF(load_activation_matrix(f"{model}_gender"),
                          normalize_rows=True)

    # Build matched-pair lookup: pF_i → pM_i and pM_i → pF_i
    matched = {}
    for pf, pm in GENDER_PAIRS:
        matched[pf] = pm
        matched[pm] = pf

    # Build a copy with distance = 1/weight on every edge.
    g_dist = g.copy()
    for u, v, data in g_dist.edges(data=True):
        w = data.get("weight", 1.0)
        g_dist[u][v]["distance"] = 1.0 / w if w > 0 else float("inf")

    records = []
    for prime in PRIMES_F + PRIMES_M:
        if prime not in g_dist.nodes() or prime not in df_norm.columns:
            continue
        counter = matched.get(prime)
        targets = TARGETS_F if prime in PRIMES_F else TARGETS_M
        for target in targets:
            if target not in g_dist.nodes() or target not in df_norm.index:
                continue
            try:
                d_w = nx.shortest_path_length(g_dist, source=prime,
                                              target=target, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            al_consistent = float(df_norm.loc[target, prime])
            # Exact paired difference: consistent prime − its matched counter prime
            if counter and counter in df_norm.columns:
                al_counter = float(df_norm.loc[target, counter])
            else:
                al_counter = 0.0
            bias = al_consistent - al_counter
            records.append({
                "network":       model,
                "prime":         prime,
                "target":        target,
                "weighted_path": d_w,
                "bias_score":    bias,
            })

    topo_rows.extend(records)
    print(f"  {model}: {len(records)} pairs, "
          f"mean weighted_path={np.mean([r['weighted_path'] for r in records]):.4f}")

    # weighted_path z-scored within network so β is comparable across models.
    df_w = pd.DataFrame(records)
    if len(df_w) >= 10:
        mu  = df_w["weighted_path"].mean()
        sd  = df_w["weighted_path"].std()
        df_w["weighted_path_z"] = (df_w["weighted_path"] - mu) / sd if sd > 0 else 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md  = smf.mixedlm("bias_score ~ weighted_path_z",
                              data=df_w, groups=df_w["prime"])
            mdf = md.fit(reml=True, method="lbfgs")
        beta = mdf.fe_params["weighted_path_z"]
        se   = mdf.bse_fe["weighted_path_z"]
        tval = mdf.tvalues["weighted_path_z"]
        pval = mdf.pvalues["weighted_path_z"]
        wlme_rows.append({
            "model":                model,
            "n_pairs":              len(df_w),
            "beta_weighted_path_z": beta,
            "SE":                   se,
            "t":                    tval,
            "p":                    pval,
            "mean_weighted_path":   mu,
            "sd_weighted_path":     sd,
            "mean_bias":            df_w["bias_score"].mean(),
        })
        print(f"    LME (z-scored): β={beta:.4f} (SE={se:.4f}), t={tval:.2f}, p={pval:.4f}")

# Save long-format table with z-scored weighted path added per network.
topo_df = pd.DataFrame(topo_rows)
topo_df["weighted_path_z"] = (
    topo_df.groupby("network")["weighted_path"]
           .transform(lambda x: (x - x.mean()) / x.std())
)
topo_out = os.path.join(DATA_DIR, "topology_bias_data.csv")
topo_df.to_csv(topo_out, index=False)
print(f"  Saved: {topo_out}  ({len(topo_df):,} rows)")

# Save LME summary.
wlme_df = pd.DataFrame(wlme_rows)
wlme_df.to_csv(
    os.path.join(ROBUSTNESS_DIR, "robustness_weighted_path_lme.csv"),
    index=False
)
print(f"  Saved: robustness_weighted_path_lme.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Pooled LME (thesis SQ2 main model)
# ═══════════════════════════════════════════════════════════════════════════════
# Pools all four networks. Additive model: shared path→bias slope, controlling
# for network. Interaction model: tests whether that slope differs by network.
print("══ Experiment 2: Pooled LME — thesis SQ2 main model ════════════════════")

# Load from the saved CSV so this section can be re-run independently.
topo_path = os.path.join(DATA_DIR, "topology_bias_data.csv")
if os.path.exists(topo_path):
    df_all = pd.read_csv(topo_path)
else:
    df_all = topo_df.copy()   # fall back to in-memory frame if CSV not written yet

print(f"  N total pairs: {len(df_all):,}  "
      f"({df_all['network'].nunique()} networks, "
      f"{df_all['prime'].nunique()} primes)")


def _fit_pooled(formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md  = smf.mixedlm(formula, data=data, groups=data["prime"])
        mdf = md.fit(reml=True, method="lbfgs")
    return mdf


def _fe_table(mdf):
    """Return fixed-effects as a tidy DataFrame with scientific-notation p-values."""
    rows = []
    for param in mdf.fe_params.index:
        rows.append({
            "parameter": param,
            "beta":      round(mdf.fe_params[param], 4),
            "SE":        round(mdf.bse_fe[param],    4),
            "z":         round(mdf.tvalues[param],    2),
            "p":         f"{mdf.pvalues[param]:.2e}",
        })
    return pd.DataFrame(rows)


# ── Main model (additive, reference = Humans) ─────────────────────────────────
print('\n── Main model: bias_score ~ weighted_path_z + C(network, Treatment("Humans")) ─')
mdf_main = _fit_pooled(
    'bias_score ~ weighted_path_z + C(network, Treatment("Humans"))',
    df_all,
)
fe_main = _fe_table(mdf_main)
print(fe_main.to_string(index=False))

main_txt = os.path.join(STATS_DIR, "lme_pooled_main.txt")
with open(main_txt, "w") as fh:
    fh.write(str(mdf_main.summary()))
    fh.write("\n\nFixed-effect coefficients\n")
    fh.write(fe_main.to_string(index=False))
print(f"\n  Saved: {main_txt}")

# ── Interaction model (reference = Humans) ────────────────────────────────────
print('\n── Interaction model: bias_score ~ weighted_path_z * C(network, Treatment("Humans")) ─')
mdf_int = _fit_pooled(
    'bias_score ~ weighted_path_z * C(network, Treatment("Humans"))',
    df_all,
)
fe_int = _fe_table(mdf_int)
print(fe_int.to_string(index=False))

# Print per-network slopes for easy reading.
base = mdf_int.fe_params["weighted_path_z"]
print(f"\n  Per-network weighted_path_z slopes (Humans slope + interaction):")
print(f"  Humans  (reference): {base:.4f}  "
      f"(p={mdf_int.pvalues['weighted_path_z']:.2e})")
for net in ["Haiku", "Llama3", "Mistral"]:
    key   = f'weighted_path_z:C(network, Treatment("Humans"))[T.{net}]'
    slope = base + mdf_int.fe_params[key]
    pval  = mdf_int.pvalues[key]
    print(f"  {net:<8}: {slope:.4f}  (interaction p={pval:.2e})")

int_txt = os.path.join(STATS_DIR, "lme_pooled_interaction.txt")
with open(int_txt, "w") as fh:
    fh.write(str(mdf_int.summary()))
    fh.write("\n\nFixed-effect coefficients\n")
    fh.write(fe_int.to_string(index=False))
print(f"\n  Saved: {int_txt}")

print("\nRobustness checks complete.")
