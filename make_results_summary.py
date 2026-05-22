"""
make_results_summary.py — consolidate all thesis Results numbers into one .txt.

Reads:
  data/topology_metrics.csv
  data/topology_comparisons.csv
  data/stats/wilcoxon_gender_{F,M}.csv
  data/topology_bias_data.csv                (re-fits the two pooled LMEs)
  data/robustness/robustness_weighted_path_lme.csv

Writes:
  results_summary.txt   (in project root)

Run:
  python3 make_results_summary.py
"""

import os
import sys
import warnings
import datetime
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MODELS, DATA_DIR, STATS_DIR, ROBUSTNESS_DIR

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_summary.txt")


# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_p(p):
    """Format p-value: 3 dp, or '<.001' if smaller."""
    if pd.isna(p):
        return "    n/a"
    if p < 0.001:
        return "  <.001"
    return f"{p:7.3f}"


def fmt_coef(x, width=8, dp=3):
    if pd.isna(x):
        return f"{'n/a':>{width}}"
    return f"{x:>{width}.{dp}f}"


def section(title, n=72):
    bar = "═" * n
    return f"\n{bar}\n  {title}\n{bar}\n"


def subsection(title, n=72):
    return f"\n{title}\n{'─' * n}\n"


def fit_pooled(formula, data):
    # Fitted by REML.  Fixed-effect inference uses Wald z-statistics, which
    # are asymptotically valid but approximate in finite samples.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, data=data,
                           groups=data["prime"]).fit(reml=True, method="lbfgs")


def fe_rows(mdf):
    """Return a list of (param, beta, se, z, p, ci_lo, ci_hi) for a fitted model."""
    rows = []
    ci = mdf.conf_int()
    for param in mdf.fe_params.index:
        rows.append((
            param,
            mdf.fe_params[param],
            mdf.bse_fe[param],
            mdf.tvalues[param],
            mdf.pvalues[param],
            ci.loc[param, 0],
            ci.loc[param, 1],
        ))
    return rows


# ── load CSVs ────────────────────────────────────────────────────────────────
topo      = pd.read_csv(os.path.join(DATA_DIR, "topology_metrics.csv"))
comp      = pd.read_csv(os.path.join(DATA_DIR, "topology_comparisons.csv"))
wil_F     = pd.read_csv(os.path.join(STATS_DIR, "wilcoxon_gender_F.csv"),  index_col=0)
wil_M     = pd.read_csv(os.path.join(STATS_DIR, "wilcoxon_gender_M.csv"),  index_col=0)
per_lme   = pd.read_csv(os.path.join(ROBUSTNESS_DIR,
                                     "robustness_weighted_path_lme.csv"))
topo_bias = pd.read_csv(os.path.join(DATA_DIR, "topology_bias_data.csv"))

topo.set_index("model", inplace=True)
per_lme.set_index("model", inplace=True)


# ── re-fit pooled LMEs ────────────────────────────────────────────────────────
mdf_main = fit_pooled(
    'bias_score ~ weighted_path_z + C(network, Treatment("Humans"))',
    topo_bias,
)
mdf_int  = fit_pooled(
    'bias_score ~ weighted_path_z * C(network, Treatment("Humans"))',
    topo_bias,
)


# ── build output text ─────────────────────────────────────────────────────────
out = []
out.append("THESIS RESULTS SUMMARY")
out.append(f"Generated: {datetime.date.today().isoformat()}")
out.append("Matrix source: Abramski et al. (2025) R/spreadr matrices "
           "(LWOW/reproducibility/data/LDT_analyses/FA_matrices/)")
out.append(f"Total prime-target observations in LME data: {len(topo_bias):,} "
           f"({topo_bias['network'].nunique()} networks, "
           f"{topo_bias['prime'].nunique()} primes)")


# ── 1. Topology metrics ───────────────────────────────────────────────────────
out.append(section("1. TOPOLOGY METRICS (SQ1)"))
out.append("Source: data/topology_metrics.csv\n")
out.append(f"{'Model':<10}  {'ASPL [95% CI]':<26}  "
           f"{'CC_unweighted [95% CI]':<28}")
out.append("─" * 68)
for m in MODELS:
    r = topo.loc[m]
    aspl = f"{r['ASPL']:.3f} [{r['ASPL_CI_lo']:.3f}, {r['ASPL_CI_hi']:.3f}]"
    ccu  = f"{r['CC_unweighted']:.3f} [{r['CC_uw_CI_lo']:.3f}, {r['CC_uw_CI_hi']:.3f}]"
    out.append(f"{m:<10}  {aspl:<26}  {ccu:<28}")


# ── 2. Topology comparisons (Mann–Whitney) ────────────────────────────────────
out.append(section("2. TOPOLOGY COMPARISONS — Mann–Whitney U  (each LLM vs Humans, two-sided)"))
out.append("Source: data/topology_comparisons.csv\n")
out.append(f"{'Comparison':<22}  "
           f"{'ASPL: U':>14}  {'p':>8}    "
           f"{'CC_uw: U':>14}  {'p':>8}")
out.append("─" * 72)
for _, row in comp.iterrows():
    out.append(
        f"{row['comparison']:<22}  "
        f"{row['ASPL_U']:>14,.0f}  {fmt_p(row['ASPL_p'])}    "
        f"{row['CC_uw_U']:>14,.0f}  {fmt_p(row['CC_uw_p'])}"
    )


# ── 3. Wilcoxon bias results ──────────────────────────────────────────────────
out.append(section("3. WILCOXON BIAS RESULTS (replication of Abramski et al., 2025)"))
out.append("Source: data/stats/wilcoxon_gender_F.csv,  wilcoxon_gender_M.csv")
out.append("One-sided (alternative='greater'); effect size r = z / √n,  n = 125\n")

for label, df in [("Female-related targets (T_F)  —  DiffF(t) = AL_pF(t) − AL_pM(t)",  wil_F),
                  ("Male-related targets (T_M)    —  DiffM(t) = AL_pM(t) − AL_pF(t)",  wil_M)]:
    out.append(subsection(label))
    out.append(f"{'Model':<10}  {'z':>9}  {'p':>8}  {'r (effect)':>11}  {'n':>5}")
    out.append("─" * 50)
    for m in MODELS:
        r = df.loc[m]
        out.append(f"{m:<10}  {r['z']:>9.3f}  {fmt_p(r['p'])}  "
                   f"{r['effect']:>11.3f}  {int(r['n']):>5}")


# ── 4. LME pooled — additive (main) ───────────────────────────────────────────
out.append(section("4. LME POOLED — MAIN MODEL (additive)"))
out.append('Formula: bias_score ~ weighted_path_z + C(network, Treatment("Humans")) + (1 | prime)')
out.append("Estimation: REML, statsmodels.mixedlm  (L-BFGS-B)")
out.append(f"N observations: {int(mdf_main.nobs):,}   |   "
           f"N primes (random groups): {len(mdf_main.random_effects)}\n")

out.append(f"{'Parameter':<50}  {'β':>8}  {'SE':>7}  {'z':>7}  {'p':>8}  "
           f"{'95% CI':>21}")
out.append("─" * 110)
for param, beta, se, z, p, lo, hi in fe_rows(mdf_main):
    ci = f"[{lo:>7.3f}, {hi:>7.3f}]"
    out.append(f"{param:<50}  {fmt_coef(beta)}  {fmt_coef(se,7)}  "
               f"{fmt_coef(z,7,2)}  {fmt_p(p)}  {ci:>21}")

# Random effect variance for prime
sig2_prime = float(mdf_main.cov_re.iloc[0, 0])
out.append(f"\nRandom effects:")
out.append(f"  σ²_prime (random intercept variance):  {sig2_prime:.4f}")
out.append(f"  σ²_resid (residual variance):          {float(mdf_main.scale):.4f}")


# ── 5. LME pooled — interaction model ─────────────────────────────────────────
out.append(section("5. LME POOLED — INTERACTION MODEL"))
out.append('Formula: bias_score ~ weighted_path_z * C(network, Treatment("Humans")) + (1 | prime)')
out.append("Estimation: REML, statsmodels.mixedlm\n")

out.append(f"{'Parameter':<60}  {'β':>8}  {'SE':>7}  {'z':>7}  {'p':>8}  "
           f"{'95% CI':>21}")
out.append("─" * 120)
for param, beta, se, z, p, lo, hi in fe_rows(mdf_int):
    ci = f"[{lo:>7.3f}, {hi:>7.3f}]"
    out.append(f"{param:<60}  {fmt_coef(beta)}  {fmt_coef(se,7)}  "
               f"{fmt_coef(z,7,2)}  {fmt_p(p)}  {ci:>21}")

sig2_prime_int = float(mdf_int.cov_re.iloc[0, 0])
out.append(f"\nRandom effects:")
out.append(f"  σ²_prime:  {sig2_prime_int:.4f}")
out.append(f"  σ²_resid:  {float(mdf_int.scale):.4f}")

# Per-network slopes derived from interaction model
base = mdf_int.fe_params["weighted_path_z"]
out.append(f"\nPer-network slopes  (Humans = reference; LLM slope = base + interaction):")
out.append(f"  Humans   slope = {base:>7.3f}   (p = {fmt_p(mdf_int.pvalues['weighted_path_z']).strip()})")
for net in ["Haiku", "Llama3", "Mistral"]:
    key   = f'weighted_path_z:C(network, Treatment("Humans"))[T.{net}]'
    slope = base + mdf_int.fe_params[key]
    out.append(f"  {net:<8} slope = {slope:>7.3f}   "
               f"(interaction p = {fmt_p(mdf_int.pvalues[key]).strip()})")


# ── 6. Per-network LMEs (supplementary) ───────────────────────────────────────
out.append(section("6. PER-NETWORK LMEs (supplementary)"))
out.append("Source: data/robustness/robustness_weighted_path_lme.csv")
out.append("One LME per network: bias_score ~ weighted_path_z + (1 | prime)\n")

out.append(f"{'Model':<10}  {'β (weighted_path_z)':>20}  {'SE':>8}  {'t':>7}  "
           f"{'p':>8}  {'n_pairs':>9}")
out.append("─" * 75)
for m in MODELS:
    r = per_lme.loc[m]
    out.append(f"{m:<10}  {r['beta_weighted_path_z']:>20.3f}  "
               f"{r['SE']:>8.3f}  {r['t']:>7.2f}  {fmt_p(r['p'])}  "
               f"{int(r['n_pairs']):>9,}")


# ── write file ────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    f.write("\n".join(out) + "\n")

print(f"Saved: {OUT_PATH}")
print(f"({sum(1 for _ in open(OUT_PATH))} lines)")
