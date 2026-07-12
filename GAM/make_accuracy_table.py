"""
Generate condensed accuracy summary table (accuracy_summary.tex).

Layout  : rows grouped by spline basis; column groups — Gaussian and Logistic,
          each reporting relative MISE.
Gaussian columns : relative MISE (global), relative MISE (local)
Logistic columns : relative MISE on linear predictor (global/local)

MISE is reported relative to gam_cr_gcv at n = 500.
Bold marks the unique best method within each basis group × column.

Outputs
-------
results/tables/accuracy_summary.tex  — LaTeX tabular (booktabs)
results/tables/accuracy_summary.md   — Markdown table

Usage
-----
    python public/make_accuracy_table.py

Requires: numpy, pandas
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = "gam_cr_gcv"

# Methods grouped by spline basis, in display order.
GROUPS = [
    {"basis": "Natural cubic", "methods": ["gam_cr_gcv", "bam_cr_gcv"]},
    {"basis": "P-splines",     "methods": ["gam_ps_gcv", "gamdist_ps_ml", "pygam_gcv"]},
    {"basis": "B-splines",     "methods": ["gam_bs_gcv", "statsmodels_bs_gcv"]},
]

ALL_METHODS = [m for g in GROUPS for m in g["methods"]]

METHOD_LABELS = {
    "gam_cr_gcv":         "mgcv::gam (cr, GCV) — baseline",
    "bam_cr_gcv":         "mgcv::bam (cr, GCV)",
    "gam_ps_gcv":         "mgcv::gam (ps, GCV)",
    "gamdist_ps_ml":      "gamlss (ps, ML)",
    "pygam_gcv":          "pyGAM (ps, GCV)",
    "gam_bs_gcv":         "mgcv::gam (bs, GCV)",
    "statsmodels_bs_gcv": "statsmodels (bs, GCV)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_task(task: str) -> pd.DataFrame:
    """Concatenate all simulation result CSVs that contain the given task."""
    frames = []
    for f in RESULTS_DIR.glob("*.csv"):
        if "benchmark" in f.name:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "task" not in df.columns:
            continue
        sub = df[df["task"] == task]
        if not sub.empty:
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Metric computation ────────────────────────────────────────────────────────

def relative_mise(df: pd.DataFrame) -> dict:
    """Per-method MISE relative to gam_cr_gcv, averaged across n."""
    df = df.copy()
    df["mise"] = pd.to_numeric(df["mise"], errors="coerce")
    base_by_n = df[df["method"] == BASELINE].groupby("n")["mise"].mean()
    result = {}
    for m in ALL_METHODS:
        mdf = df[df["method"] == m].groupby("n")["mise"].mean()
        ratios = [
            mdf[n] / base_by_n[n]
            for n in mdf.index
            if n in base_by_n.index and pd.notna(mdf[n]) and base_by_n[n] > 0
        ]
        result[m] = float(np.mean(ratios)) if ratios else float("nan")
    return result


# ── Build data frame ──────────────────────────────────────────────────────────

N_FILTER = 500   # report metrics at this training size

def _filter_n(df): return df[df["n"] == N_FILTER] if not df.empty else df

global_df       = _filter_n(load_task("global"))
local_df        = _filter_n(load_task("local"))
logi_global_df  = _filter_n(load_task("logistic_global"))
logi_local_df   = _filter_n(load_task("logistic_local"))

if global_df.empty and local_df.empty:
    print("No simulation data found — run simulations first.")
    raise SystemExit(1)

def _rel(df): return relative_mise(df) if not df.empty else {}
nan = float("nan")

tbl = pd.DataFrame({
    # Gaussian univariate
    "g_mise_global":  [_rel(global_df).get(m, nan)  for m in ALL_METHODS],
    "g_mise_local":   [_rel(local_df).get(m, nan)   for m in ALL_METHODS],
    # Logistic
    "l_mise_global":  [_rel(logi_global_df).get(m, nan) for m in ALL_METHODS],
    "l_mise_local":   [_rel(logi_local_df).get(m, nan)  for m in ALL_METHODS],
}, index=ALL_METHODS)

GAUSS_COLS = ["g_mise_global", "g_mise_local"]
LOGI_COLS  = ["l_mise_global", "l_mise_local"]
ALL_COLS   = GAUSS_COLS + LOGI_COLS

def best_in_group(group_methods, col):
    """Unique lowest MISE within the group; None if tied or all missing."""
    vals = tbl.loc[[m for m in group_methods if m in tbl.index], col].dropna()
    if vals.empty:
        return None
    winners = vals[vals == vals.min()]
    return winners.index[0] if len(winners) == 1 else None


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt_md(v, bold=False):
    if np.isnan(v):
        return "—"
    s = f"{v:.2f}"
    return f"**{s}**" if bold else s


def fmt_tex(v, bold=False):
    if np.isnan(v):
        return r"\text{---}"
    s = f"{v:.3f}"
    return rf"\mathbf{{{s}}}" if bold else s


# ── Markdown table ────────────────────────────────────────────────────────────

header = ("| Basis | Method "
          "| MISE glob. | MISE loc. "
          "| MISE LP glob. | MISE LP loc. |")
sep    = ("|:------|:-------"
          "|:-----------:|:---------:"
          "|:-------------:|:------------:|")

rows_md = [header, sep]

for g in GROUPS:
    basis, methods = g["basis"], g["methods"]
    best = {c: best_in_group(methods, c) for c in ALL_COLS}

    for i, m in enumerate(methods):
        if m not in tbl.index:
            continue
        r = tbl.loc[m]
        cells = [
            basis if i == 0 else "",
            METHOD_LABELS.get(m, m),
            fmt_md(r["g_mise_global"], bold=(best["g_mise_global"] == m)),
            fmt_md(r["g_mise_local"],  bold=(best["g_mise_local"]  == m)),
            fmt_md(r["l_mise_global"], bold=(best["l_mise_global"] == m)),
            fmt_md(r["l_mise_local"],  bold=(best["l_mise_local"]  == m)),
        ]
        rows_md.append("| " + " | ".join(cells) + " |")

md_table = "\n".join(rows_md)

md_out = TABLES_DIR / "accuracy_summary.md"
md_out.write_text(md_table + "\n")
print(f"saved {md_out}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

# Shorter labels for the narrow Method column.
METHOD_LABELS_TEX = {
    "gam_cr_gcv":         r"gam (cr) $\star$",
    "bam_cr_gcv":         r"bam (cr)",
    "gam_ps_gcv":         r"gam (ps)",
    "gamdist_ps_ml":      r"gamlss",
    "pygam_gcv":          r"pyGAM",
    "gam_bs_gcv":         r"gam (bs)",
    "statsmodels_bs_gcv": r"statsmodels",
}

def label_tex(m):
    return METHOD_LABELS_TEX.get(m, m)

tex_lines = [
    r"\begin{table}[ht]",
    r"\centering",
    (rf"\caption{{GAM accuracy summary at $n = {N_FILTER}$."
     r" Each cell reports MISE relative to \texttt{gam\_cr\_gcv} ($M$)"
     r" for univariate GAMs with a global ($g$) or local ($l$) smooth,"
     r" under Gaussian and logistic (linear-predictor scale) response."
     r" Methods are grouped by spline basis. Bold = unique best within group.}"),
    r"\label{tab:accuracy_summary}",
    r"\resizebox{\linewidth}{!}{%",
    r"\begin{tabular}{l p{2.1cm} rr rr}",
    r"\toprule",
    r" & & \multicolumn{2}{c}{Gaussian} & \multicolumn{2}{c}{Logistic} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    ("Basis & Method"
     r" & $M_g$ & $M_l$"
     r" & $M^\text{LP}_g$ & $M^\text{LP}_l$ \\"),
    r"\midrule",
]

for gi, g in enumerate(GROUPS):
    basis, methods = g["basis"], g["methods"]
    best = {c: best_in_group(methods, c) for c in ALL_COLS}

    for i, m in enumerate(methods):
        if m not in tbl.index:
            continue
        r = tbl.loc[m]
        cells = [
            basis if i == 0 else "",
            label_tex(m),
            f"${fmt_tex(r['g_mise_global'], bold=(best['g_mise_global'] == m))}$",
            f"${fmt_tex(r['g_mise_local'],  bold=(best['g_mise_local']  == m))}$",
            f"${fmt_tex(r['l_mise_global'], bold=(best['l_mise_global'] == m))}$",
            f"${fmt_tex(r['l_mise_local'],  bold=(best['l_mise_local']  == m))}$",
        ]
        tex_lines.append(" & ".join(cells) + r" \\")

    if gi < len(GROUPS) - 1:
        tex_lines.append(r"\midrule")

tex_lines += [
    r"\bottomrule",
    r"\end{tabular}}",  # closes \resizebox
    (r"\par\smallskip"
     r"{\footnotesize $\star$: baseline (reference MISE = 1). Bold = unique best within basis group.}"),
    r"\end{table}",
]

tex_out = TABLES_DIR / "accuracy_summary.tex"
tex_out.write_text("\n".join(tex_lines) + "\n")
print(f"saved {tex_out}")

print()
print(tbl.round(3).to_string())
