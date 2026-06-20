"""
Generate scalability table (table_scalability.tex).

Reports median elapsed time (s) and algorithmic success rate for each
(scenario, method, n, p) cell, with Gaussian additive and logistic scenarios
side-by-side. Methods are grouped by spline basis. Bold marks the
statistically best method within each group (paired Wilcoxon signed-rank
test vs. all competitors; Bonferroni-corrected alpha = 0.05).

Reads
-----
results/r_benchmark.csv           — R benchmark results
results/python_benchmark.csv      — Python benchmark results
results/benchmark_chunks/*.csv    — per-cell chunk files (if present)

Output
------
results/tables/table_scalability.tex

Usage
-----
    python public/make_scalability_table.py

Requires: numpy, pandas, scipy
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "scipy"]
# ///

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Methods grouped by spline basis — mirrors accuracy_summary.tex.
BASIS_GROUPS = [
    ("Natural cubic", ["gam_cr", "bam_cr"]),
    ("P-splines",     ["gam_ps", "gamdist_ps", "pygam"]),
    ("B-splines",     ["gam_bs", "statsmodels_bs"]),
]
LABELS = {
    "gam_cr": "gam (cr)", "bam_cr": "bam (cr)", "gam_ps": "gam (ps)",
    "gamdist_ps": "gamlss", "gam_bs": "gam (bs)", "pygam": "pyGAM",
    "statsmodels_bs": "statsmodels",
}

# Scenario family -> column-group header.
FAMILIES = [("additive", "Gaussian"), ("logistic", "Logistic")]

METRICS = [("elapsed_s", "Elapsed time (s)"), ("mem_mb", "Peak memory (MB)")]

CENSOR = r"$\text{---}$"   # no completed run


def _load_bench() -> pd.DataFrame | None:
    """Combine summary CSVs and per-chunk CSVs, de-duplicating on the run key."""
    frames = []
    for fname in ("r_benchmark.csv", "python_benchmark.csv"):
        p = RESULTS_DIR / fname
        if p.exists():
            frames.append(pd.read_csv(p))
    chunk_dir = RESULTS_DIR / "benchmark_chunks"
    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"  warning: could not read {f.name}: {e}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    key = [c for c in ("scenario", "method", "n", "rep") if c in df.columns]
    if key:
        df = df.drop_duplicates(subset=key, keep="first")
    for c in ("elapsed_s", "mem_mb"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").astype("Int64")
    return df


def _fmt(x: float) -> str:
    if x is None or pd.isna(x):
        return CENSOR
    if x >= 1000:
        return f"{x:.0f}"
    if x >= 100:
        return f"{x:.1f}"
    if x >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _discover(agg: pd.DataFrame):
    """Return (families, ns, ps, groups) actually present, in canonical order."""
    sc = agg["scenario"].astype(str)
    fam_present = set(sc.str.replace(r"_p\d+$", "", regex=True))
    families = [(f, lab) for f, lab in FAMILIES if f in fam_present]
    ns = sorted(int(n) for n in agg["n"].dropna().unique())
    ps = sorted({int(s.split("_p")[-1]) for s in sc if "_p" in s})
    present = set(agg["method"])
    groups = [(b, [m for m in ms if m in present]) for b, ms in BASIS_GROUPS]
    groups = [(b, ms) for b, ms in groups if ms]
    return families, ns, ps, groups


def _cell(idx, fam, p, n, m, vcol):
    try:
        v = idx.loc[(f"{fam}_p{p:02d}", m, n), vcol]
    except KeyError:
        return np.nan
    return v.iloc[0] if isinstance(v, pd.Series) else v


def _isolated_marks(idx, families, ns, ps, groups):
    """Flag completed cells immediately following a censored cell (gap in the
    p-axis run). The figure draws no connecting line through these points."""
    marks = {}
    methods = [m for _, ms in groups for m in ms]
    for fam, _ in families:
        for vcol, _ in METRICS:
            for m in methods:
                for n in ns:
                    done = [pd.notna(_cell(idx, fam, p, n, m, vcol)) for p in ps]
                    for i in range(1, len(ps)):
                        if done[i] and not done[i - 1]:
                            marks[(fam, vcol, m, n, ps[i])] = True
    return marks


def _best_marks_wilcox(raw, families, ns, ps, groups, metrics,
                       alpha=0.05, min_pairs=10):
    """Bold = significantly best within basis group by paired Wilcoxon
    signed-rank test (pairing = same rep index).

    For each (vcol, fam, n, p, basis_group):
      1. Find the method with the lowest median over completed reps.
      2. Compare it to every other method in the group with data, using reps
         where both methods completed (paired design).
      3. Apply Bonferroni correction across the (k-1) comparisons.
      4. Bold only if significantly smaller than ALL others at corrected alpha.
         If any comparison is non-significant or has too few paired reps, no
         method is bolded for that cell.
    """
    best = set()
    raw = raw.copy()
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce")

    for vcol, _ in metrics:
        if vcol not in raw.columns:
            continue
        for fam, _ in families:
            for n in ns:
                for basis, methods in groups:
                    for p in ps:
                        scenario = f"{fam}_p{p:02d}"
                        rep_data = {}
                        for m in methods:
                            sub = raw.loc[
                                (raw["scenario"] == scenario) &
                                (raw["method"] == m) &
                                (raw["n"] == n),
                                ["rep", vcol]
                            ].dropna(subset=[vcol])
                            if not sub.empty:
                                rep_data[m] = (sub.set_index("rep")[vcol]
                                               .groupby(level=0).median())
                        if len(rep_data) < 2:
                            continue
                        medians = {m: s.median() for m, s in rep_data.items()}
                        best_m = min(medians, key=medians.get)
                        others = [m for m in rep_data if m != best_m]
                        alpha_corr = alpha / len(others)   # Bonferroni
                        all_sig = True
                        for other in others:
                            # Skip if the formatted strings are identical
                            # (difference too small to show in the table).
                            if _fmt(medians[best_m]) == _fmt(medians[other]):
                                all_sig = False
                                break
                            common = rep_data[best_m].index.intersection(
                                rep_data[other].index)
                            if len(common) < min_pairs:
                                all_sig = False
                                break
                            diff = (rep_data[best_m][common].values
                                    - rep_data[other][common].values)
                            if (diff == 0).all():
                                all_sig = False
                                break
                            try:
                                _, p_val = stats.wilcoxon(
                                    diff, alternative="less")
                                if p_val >= alpha_corr:
                                    all_sig = False
                                    break
                            except Exception:
                                all_sig = False
                                break
                        if all_sig:
                            best.add((vcol, fam, n, basis, best_m, p))
    return best


def _subtable_combined(idx, families, ns, ps, groups, marks, best,
                       success_rates) -> str:
    """Build the LaTeX tabular body combining elapsed time and success rate
    in a single cell (format: 'time / success%' when success < 100%)."""
    np_ = len(ps)
    ncol = 2 + len(families) * np_
    colspec = "l l " + " ".join(["r" * np_] * len(families))

    grp = " & ".join(rf"\multicolumn{{{np_}}}{{c}}{{{lab}}}"
                     for _, lab in families)
    cmids = []
    start = 3
    for _ in families:
        cmids.append(rf"\cmidrule(lr){{{start}-{start + np_ - 1}}}")
        start += np_
    pcols = " & ".join([rf"$p{{=}}{p}$" for _ in families for p in ps])

    lines = [
        r"\textbf{Elapsed time (s) / Algorithmic success rate}\par\smallskip",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf" & & {grp} \\",
        "".join(cmids),
        rf"Basis & Method & {pcols} \\",
        r"\midrule",
    ]

    for ni, n in enumerate(ns):
        if ni > 0:
            lines.append(r"\midrule")
        nlabel = f"{n:,}".replace(",", r"{,}")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{$n = {nlabel}$}}}}\\")
        lines.append(r"\midrule")
        for gi, (basis, methods) in enumerate(groups):
            for mi, m in enumerate(methods):
                cells = []
                for fam, _ in families:
                    for p in ps:
                        t = _cell(idx, fam, p, n, m, "elapsed_s")
                        sr = success_rates.get((fam, p, n, m), 0)
                        t_str = _fmt(t)
                        # Show success rate only when < 100%
                        sr_str = f"{sr}%" if sr < 100 else ""
                        if marks.get((fam, "elapsed_s", m, n, p)) and t_str != CENSOR:
                            t_str = rf"${t_str}^\dagger$"
                        if (("elapsed_s", fam, n, basis, m, p) in best and
                                t_str != CENSOR):
                            t_str = rf"\textbf{{{t_str}}}"
                        s = t_str + (rf" / {sr_str}" if sr_str else "")
                        cells.append(s)
                bcol = basis if mi == 0 else ""
                row = " & ".join([bcol, LABELS.get(m, m)] + cells)
                lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def main():
    raw = _load_bench()
    if raw is None:
        print("no benchmark data found — nothing to generate")
        return

    metric_cols = [c for c, _ in METRICS if c in raw.columns]
    agg = (raw.groupby(["scenario", "method", "n"])[metric_cols]
              .median().reset_index())
    idx = agg.set_index(["scenario", "method", "n"]).sort_index()

    families, ns, ps, groups = _discover(agg)
    ps = [p for p in ps if p != 5]   # omit p=5 to keep the table compact
    metrics = [(c, h) for c, h in METRICS if c in metric_cols]
    print(f"  families={[f for f, _ in families]}  n={ns}  p={ps}")
    print(f"  basis groups={[(b, ms) for b, ms in groups]}")

    marks = _isolated_marks(idx, families, ns, ps, groups)

    # Bold = statistically best (Wilcoxon signed-rank, Bonferroni-corrected).
    print("  running Wilcoxon tests …")
    best_wx = _best_marks_wilcox(raw, families, ns, ps, groups, metrics)

    # Algorithmic success rate: count of completed reps per (fam, p, n, m).
    success_rates = {}
    for fam, _ in families:
        for p in ps:
            for n in ns:
                for _, methods in groups:
                    for m in methods:
                        sc = f"{fam}_p{p:02d}"
                        sub = raw.loc[
                            (raw["scenario"] == sc) &
                            (raw["method"] == m) &
                            (raw["n"] == n),
                            "elapsed_s"
                        ].dropna()
                        success_rates[(fam, p, n, m)] = len(sub)

    caption = (
        r"Scalability of GAM fitting versus the number of predictors $p$: "
        r"median elapsed time (s) and algorithmic success rate over replicates "
        r"(displayed as \texttt{time~/$\,$success}), for Gaussian (additive) "
        r"and logistic scenarios at each sample size $n$. Methods are grouped by "
        r"spline basis. \textbf{Bold} = significantly best within basis group "
        r"(paired Wilcoxon signed-rank test; Bonferroni-corrected $\alpha = 0.05$); "
        r"``" + CENSOR + r"'' marks a setting with no completed run; $^\dagger$ "
        r"marks a gapped completion (see text). Success rate shown only when "
        r"$< 100\%$ (e.g., \texttt{4.98 / 84\%} = 4.98\,s with 84 of 100 reps "
        r"completed).")

    subtable = _subtable_combined(
        idx, families, ns, ps, groups, marks, best_wx, success_rates)

    blocks = [
        "% Auto-generated by public/make_scalability_table.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{amsmath}, "
        "\\usepackage{graphicx}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:scalability}",
        subtable,
        r"\end{table}",
        "",
    ]
    out = TABLES_DIR / "table_scalability.tex"
    out.write_text("\n".join(blocks))
    print(f"saved {out}")
    print(f"  {len(best_wx)} bold cells (Wilcoxon)")


if __name__ == "__main__":
    main()
