"""
02_bias_analysis.py — Gender stereotype bias evaluation.

Replicates Section 3 ("Stereotypes related to gender") of Abramski et al.
(2025) using the activation matrices released with that paper (the R/spreadr
matrices under LWOW/reproducibility/data/LDT_analyses/FA_matrices/).

Methodology (stereotypes approach)
-----------------------------------
1. Normalise each activation matrix: L2-normalise columns then rows.
2. For every female-related prime / male-related prime pair (p_F, p_M) and
   every female-related target t_F:
       Diff_F(t) = AL_{p_F}(t) − AL_{p_M}(t)
   and for every male-related target t_M:
       Diff_M(t) = AL_{p_M}(t) − AL_{p_F}(t)
   Positive values → stereotype-consistent activation.
3. Wilcoxon signed-rank test on the 125 paired differences (5 pairs × 25
   targets) to test whether stereotype-consistent activation exceeds chance.
   Effect size = z / sqrt(n).

Output
------
  data/stats/wilcoxon_gender_F.csv
  data/stats/wilcoxon_gender_M.csv

Run:
  python3 02_bias_analysis.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from functools import reduce

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    MODELS,
    GENDER_PAIRS, PRIMES_F, PRIMES_M,
    TARGETS_F, TARGETS_M,
    STATS_DIR,
    load_activation_matrix, normalizeDF, matricesGender,
)

os.makedirs(STATS_DIR, exist_ok=True)


# ── 1. Load activation matrices and normalise ─────────────────────────────────
# Normalisation: L2 by column (controls for node centrality), then L2 by row
# (controls for differences in overall activation level across primes).
print("Loading and normalising activation matrices …")

gender_dfs      = {}   # raw activation matrices
gender_dfs_norm = {}   # normalised matrices

for model in MODELS:
    df = load_activation_matrix(f"{model}_gender")
    gender_dfs[model] = df

    # normalizeDF expects a 'node' column and normalises the rest.
    df_norm = normalizeDF(df, normalize_rows=True)
    gender_dfs_norm[model] = df_norm
    print(f"  {model}: {df.shape[0]:,} nodes × {df.shape[1]-1} primes")


# 2. Filter only targets present in all four networks 
print("\nFinding node intersection across all networks …")
all_nodes = [set(gender_dfs_norm[m].index) for m in MODELS]
node_intersection = list(reduce(lambda a, b: a & b, all_nodes))
print(f"  Shared nodes: {len(node_intersection):,}")

# Filter targets to those present in the intersection.
tgts_F = [t for t in TARGETS_F if t in node_intersection]
tgts_M = [t for t in TARGETS_M if t in node_intersection]
print(f"  Female-related targets in intersection: {len(tgts_F)}/25")
print(f"  Male-related targets in intersection:   {len(tgts_M)}/25")

missing_F = [t for t in TARGETS_F if t not in node_intersection]
missing_M = [t for t in TARGETS_M if t not in node_intersection]
if missing_F:
    print(f"  [warn] Missing female targets: {missing_F}")
if missing_M:
    print(f"  [warn] Missing male targets:   {missing_M}")


# ── 3. Build target × prime activation sub-matrices ──────────────────────────
# matF: female-related targets × all 10 gender primes
# matM: male-related targets   × all 10 gender primes
print("\nExtracting target activation sub-matrices …")
mat_gender = {}
for model in MODELS:
    df_norm = gender_dfs_norm[model]
    # Keep only primes that are actually columns in the matrix.
    avail_primes_F = [p for p in PRIMES_F if p in df_norm.columns]
    avail_primes_M = [p for p in PRIMES_M if p in df_norm.columns]

    matF = matricesGender(df_norm, tgts_F, avail_primes_F, avail_primes_M)
    matM = matricesGender(df_norm, tgts_M, avail_primes_F, avail_primes_M)
    mat_gender[model] = {
        "Female-related targets": matF,
        "Male-related targets":   matM,
        "primes_F": avail_primes_F,
        "primes_M": avail_primes_M,
    }


# ── 4. Wilcoxon tests & effect sizes ─────────────────────────────────────────
# DiffF(t) = AL_pF(t) − AL_pM(t)  ∀ t ∈ TF, (pF, pM) ∈ P  (expect > 0)
# DiffM(t) = AL_pM(t) − AL_pF(t)  ∀ t ∈ TM, (pF, pM) ∈ P  (expect > 0)
# The Wilcoxon test uses 'approx' method for the z-statistic needed for effect size.
print("\n─── Wilcoxon signed-rank tests ───────────────────────────────────────")

wilcoxon_F   = {}
wilcoxon_M   = {}
al_diffs     = {}   # stored for histogram plots

for model in MODELS:
    pF = mat_gender[model]["primes_F"]
    pM = mat_gender[model]["primes_M"]

    # --- DiffF(t) = AL_pF(t) − AL_pM(t),  t ∈ TF ---
    data1_F = mat_gender[model]["Female-related targets"][pF].values.flatten()
    data2_F = mat_gender[model]["Female-related targets"][pM].values.flatten()
    diff_F  = data1_F - data2_F   # AL_pF(t) − AL_pM(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_F = stats.wilcoxon(diff_F, alternative="greater", method="approx")
    p_F     = res_F.pvalue
    z_F     = res_F.zstatistic
    eff_F   = z_F / np.sqrt(len(diff_F))
    wilcoxon_F[model] = {"p": p_F, "z": z_F, "effect": eff_F, "n": len(diff_F)}

    # --- DiffM(t) = AL_pM(t) − AL_pF(t),  t ∈ TM ---
    data1_M = mat_gender[model]["Male-related targets"][pM].values.flatten()
    data2_M = mat_gender[model]["Male-related targets"][pF].values.flatten()
    diff_M  = data1_M - data2_M   # AL_pM(t) − AL_pF(t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_M = stats.wilcoxon(diff_M, alternative="greater", method="approx")
    p_M     = res_M.pvalue
    z_M     = res_M.zstatistic
    eff_M   = z_M / np.sqrt(len(diff_M))
    wilcoxon_M[model] = {"p": p_M, "z": z_M, "effect": eff_M, "n": len(diff_M)}

    al_diffs[model] = {"Female-related targets": diff_F,
                       "Male-related targets":   diff_M}

    sig_F = "***" if p_F < 0.001 else ("**" if p_F < 0.01 else ("*" if p_F < 0.05 else "ns"))
    sig_M = "***" if p_M < 0.001 else ("**" if p_M < 0.01 else ("*" if p_M < 0.05 else "ns"))
    print(f"  {model}:  DiffF effect={eff_F:.3f} {sig_F}  |  DiffM effect={eff_M:.3f} {sig_M}")

pd.DataFrame(wilcoxon_F).T.to_csv(os.path.join(STATS_DIR, "wilcoxon_gender_F.csv"))
pd.DataFrame(wilcoxon_M).T.to_csv(os.path.join(STATS_DIR, "wilcoxon_gender_M.csv"))


print("\nBias analysis complete.")
