"""
utils.py — shared constants, paths, and helper functions for the thesis pipeline.

"""

import sys
import os
import numpy as np
import pandas as pd

# FA_Functions from the LWOW reproducibility repo — import the helpers used here.
_LWOW_REPRO = os.path.join(os.path.dirname(__file__), "LWOW", "reproducibility")
if _LWOW_REPRO not in sys.path:
    sys.path.insert(0, _LWOW_REPRO)

from FA_Functions import (  # noqa: E402
    edgelist2graph,
    normalizeDF,
    matricesGender,
)

# ─────────────────────────────────────────────────────────────────────────────
# Directory / file paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
LWOW_REPO        = os.path.join(BASE_DIR, "LWOW", "reproducibility")

# Our output directories (separate from the LWOW repo).
DATA_DIR         = os.path.join(BASE_DIR, "data")
EDGELIST_DIR     = os.path.join(DATA_DIR, "graphs", "edge_lists")
STATS_DIR        = os.path.join(DATA_DIR, "stats")
ROBUSTNESS_DIR   = os.path.join(DATA_DIR, "robustness")
FIGURES_DIR      = os.path.join(BASE_DIR, "figures")

# Activation matrices released by Abramski et al. (2025), loaded directly —
# spreading activation is not recomputed here.
R_MATRICES_DIR   = os.path.join(
    LWOW_REPO, "data", "LDT_analyses", "FA_matrices"
)

# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────
# The four sources compared in the paper (and the thesis).
MODELS = ["Humans", "Mistral", "Llama3", "Haiku"]

# Colours match the original LWOW paper figures.
MODEL_COLORS = {
    "Humans": "#3269AF",
    "Mistral": "#F19542",
    "Llama3": "#73BC6B",
    "Haiku": "#6A54A6",
}

MODEL_COLOR_PALETTES = {
    "Humans": "Blues",
    "Mistral": "Oranges",
    "Llama3": "Greens",
    "Haiku": "Purples",
}

# ─────────────────────────────────────────────────────────────────────────────
# Gender primes (Table 2 of Abramski et al., 2025)
# Each tuple is (female-related prime, male-related prime).
# ─────────────────────────────────────────────────────────────────────────────
GENDER_PAIRS = [
    ("woman",    "man"),
    ("female",   "male"),
    ("mother",   "father"),
    ("girl",     "boy"),
    ("feminine", "masculine"),
]
PRIMES_F = [p[0] for p in GENDER_PAIRS]   # female-related primes
PRIMES_M = [p[1] for p in GENDER_PAIRS]   # male-related primes
GENDER_PRIMES = PRIMES_F + PRIMES_M       # all 10 gender prime words

# Gender stereotype target adjectives (Table 3 of Abramski et al., 2025).
# 25 female-related and 25 male-related stereotypical adjectives.
TARGETS_F = [
    "affectionate", "cheerful", "compassionate", "considerate", "cooperative",
    "emotional", "empathetic", "gentle", "honest", "kind",
    "loyal", "modest", "nagging", "nurturing", "pleasant",
    "polite", "quiet", "sensitive", "submissive", "supportive",
    "sympathetic", "tender", "trusting", "understanding", "warm",
]

TARGETS_M = [
    "active", "aggressive", "ambitious", "analytical", "assertive",
    "athletic", "competitive", "confident", "courageous", "decisive",
    "determined", "dominant", "forceful", "greedy", "hostile",
    "impulsive", "independent", "intellectual", "leader", "logical",
    "outspoken", "persistent", "reckless", "stubborn", "superior",
]

# ─────────────────────────────────────────────────────────────────────────────
# Graph I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_edgelist(model, directed=False):
    """Load a filtered network edge list as a networkx (Di)Graph."""
    path = os.path.join(EDGELIST_DIR, f"FA_{model}_edgelist.csv")
    return edgelist2graph(path, directed=directed)


def load_activation_matrix(name):
    """Load an Abramski et al. (2025) R/spreadr activation matrix by file stem
    (e.g. 'Humans_gender'); first column renamed to 'node'."""
    path = os.path.join(R_MATRICES_DIR, f"{name}.csv")
    df   = pd.read_csv(path)
    if df.columns[0] != "node":
        df = df.rename(columns={df.columns[0]: "node"})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Topology helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_metric(values, n_boot=1000, seed=42):
    """Non-parametric bootstrap of the mean; returns (mean, ci_lo, ci_hi) at 95%."""
    rng    = np.random.default_rng(seed)
    arr    = np.array(values)
    boots  = [np.mean(rng.choice(arr, size=len(arr), replace=True))
              for _ in range(n_boot)]
    return (np.mean(arr),
            np.percentile(boots, 2.5),
            np.percentile(boots, 97.5))
