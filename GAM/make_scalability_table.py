"""
Generate scalability table (table_scalability.tex / .md).

Reports, in one table, median elapsed time (s), peak RSS (MB) and iterations
per second (as "time / RSS / it/s") plus the algorithmic success rate for each
(scenario, method, n, p) cell, with Gaussian additive and logistic scenarios
side by side.
Methods are grouped by spline basis. Bold marks the
unique best (median) method within each group, per metric.

Reads
-----
results/r_benchmark.csv           — R benchmark results
results/python_benchmark.csv      — Python benchmark results
results/benchmark_chunks/*.csv    — per-cell chunk files (if present)

Output
------
results/tables/table_scalability.tex   — LaTeX (booktabs, amsmath, adjustbox)
results/tables/table_scalability.md    — Markdown

Usage
-----
    python public/make_scalability_table.py

Requires: numpy, pandas
"""
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy"]
# ///

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Methods grouped by spline basis — mirrors accuracy_summary.tex.
BASIS_GROUPS = [
    ("Natural cubic", ["gam_cr", "bam_cr", "gamdist_cr"]),
    ("P-splines",     ["gam_ps", "gamlss_ps", "pygam"]),
    ("B-splines",     ["gam_bs", "statsmodels_bs"]),
]
LABELS = {
    "gam_cr": "gam (cr)", "bam_cr": "bam (cr)", "gam_ps": "gam (ps)",
    "gamlss_ps": "gamlss", "gam_bs": "gam (bs)", "pygam": "pyGAM",
    "statsmodels_bs": "statsmodels", "gamdist_cr": "gamdist",
}
# Programming environment of each method (the paper preamble defines \Ricon, \pyicon).
PYTHON_METHODS = {"pygam", "statsmodels_bs", "gamdist_cr"}


def _label(m, md=False):
    """Method name followed by its environment mark (R or Python)."""
    py = m in PYTHON_METHODS
    if md:
        return f"{LABELS.get(m, m)} {'[py]' if py else '[R]'}"
    return rf"{LABELS.get(m, m)}\,{'\\pyicon' if py else '\\Ricon'}{{}}"
# Results written before the rename used "gamdist_ps" for the gamlss fits.
LEGACY_METHODS = {"gamdist_ps": "gamlss_ps"}

# Scenario family -> column-group header.
FAMILIES = [("additive", "Gaussian"), ("logistic", "Logistic")]

# (column, table title, better direction, label suffix) — one table per metric.
METRICS = [("elapsed_s",   "Elapsed time (s) / Algorithmic success rate", "min", "time"),
           ("max_rss_mb",  "Peak RSS (MB)",                               "min", "rss"),
           ("itr_per_sec", "Iterations per second",                       "max", "its")]

CENSOR = "--"   # no completed run (en dash)


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
    df["method"] = df["method"].replace(LEGACY_METHODS)
    # pyGAM and statsmodels rows written before the common edf grid (no grid_ver = 2)
    # used non-comparable grids; ignore them.
    if "grid_ver" not in df.columns:
        df["grid_ver"] = np.nan
    df = df[~(df["method"].isin(["pygam", "statsmodels_bs"]) & (df["grid_ver"] != 2))]
    key = [c for c in ("scenario", "method", "n", "rep") if c in df.columns]
    if key:
        df = df.drop_duplicates(subset=key, keep="first")
    for c in ("elapsed_s", "itr_per_sec", "total_time_s", "max_rss_mb"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce").astype("Int64")
    return df


SCI_BELOW = 1e-3   # |x| below this is shown in scientific notation
ITS_SCALE_P = 50   # for p >= this, it/s is reported in units of 10^-3 (value x 1000)
CELL_W = "4.0cm"   # fixed width of a metric cell, so both panels share one layout
GIVEN_UP_AFTER = 5  # no completed rep after this many attempts: shown as "-- (0%)"
                    # (matches MAX_METHOD_TIMEOUTS in the benchmark scripts)
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _fmt(x: float, sig: int = 3, md: bool = False) -> str:
    """Round to `sig` significant figures: plain notation, or scientific
    (a.bc x 10^k; LaTeX math unless md) when |x| < SCI_BELOW."""
    if x is None or pd.isna(x):
        return CENSOR
    if x == 0:
        return "0"
    if abs(x) < SCI_BELOW:
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        exp = int(exp)
        return f"{mant}×10{str(exp).translate(_SUP)}" if md else rf"${mant}\times10^{{{exp}}}$"
    decimals = sig - 1 - int(np.floor(np.log10(abs(x))))
    x = round(x, decimals)
    if x != 0 and int(np.floor(np.log10(abs(x)))) != sig - 1 - decimals:
        decimals -= 1   # rounding carried into a new digit (e.g. 9.995 -> 10.0)
    return f"{x:.{max(decimals, 0)}f}"


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
        for vcol, *_ in METRICS:
            for m in methods:
                for n in ns:
                    done = [pd.notna(_cell(idx, fam, p, n, m, vcol)) for p in ps]
                    for i in range(1, len(ps)):
                        if done[i] and not done[i - 1]:
                            marks[(fam, vcol, m, n, ps[i])] = True
    return marks


def _best_marks_unique(raw, families, ns, ps, groups, metrics):
    """Bold = unique best median within the basis group (lowest, or highest for
    metrics whose direction is "max"). For each (vcol, fam, n, p, basis_group)
    the median over completed reps is compared across the methods with data;
    no method is bolded if the best value ties another method's (identical
    formatted value) or fewer than two methods have data.
    """
    best = set()
    raw = raw.copy()
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce")

    for vcol, _, direction, _ in metrics:
        if vcol not in raw.columns:
            continue
        for fam, _ in families:
            for n in ns:
                for basis, methods in groups:
                    for p in ps:
                        scenario = f"{fam}_p{p:02d}"
                        medians = {}
                        for m in methods:
                            v = raw.loc[
                                (raw["scenario"] == scenario) &
                                (raw["method"] == m) &
                                (raw["n"] == n),
                                vcol
                            ].dropna()
                            if not v.empty:
                                medians[m] = v.median()
                        if len(medians) < 2:
                            continue
                        pick = max if direction == "max" else min
                        best_m = pick(medians, key=medians.get)
                        if any(_fmt(medians[best_m]) == _fmt(medians[o])
                               for o in medians if o != best_m):
                            continue
                        best.add((vcol, fam, n, basis, best_m, p))
    return best


def _cell_text(idx, fam, p, n, m, basis, vcol, marks, best, success_rates,
               md=False):
    """Formatted value for one metric cell, in LaTeX or Markdown."""
    censor = "–" if md else CENSOR
    val = _cell(idx, fam, p, n, m, vcol)
    if vcol == "itr_per_sec" and p >= ITS_SCALE_P:
        val = val * 1e3
    s_ = _fmt(val, md=md)
    if s_ == CENSOR:
        return censor
    sci = s_.startswith("$")   # LaTeX scientific notation is already math
    if marks.get((fam, vcol, m, n, p)):
        if md:
            s_ = f"{s_}†"
        else:
            s_ = "${" + s_[1:-1] + r"}^{\dagger}$" if sci else rf"${s_}^\dagger$"
    if (vcol, fam, n, basis, m, p) in best:
        if md:
            s_ = f"**{s_}**"
        else:
            # math (scientific notation or a dagger) needs \boldmath to turn bold
            s_ = rf"\textbf{{\boldmath{s_}}}" if "$" in s_ else rf"\textbf{{{s_}}}"
    return s_


def _value(idx, fam, p, n, m, basis, vcols, marks, best, success_rates, md):
    """One cell: the metrics in `vcols` joined by ' / ' (each part bolded on its
    own), then the success rate in parentheses when below 100%. A dash when no
    replicate completed."""
    sr, att = success_rates.get((fam, p, n, m), (0, 0))
    if sr == 0:
        # nothing completed: "-- (0%)" once we have given up, else not yet decided
        gave_up = att >= GIVEN_UP_AFTER
        if md:
            return "– (0%)" if gave_up else "–"
        return rf"{CENSOR} (0\%)" if gave_up else CENSOR
    v = " / ".join(_cell_text(idx, fam, p, n, m, basis, c, marks, best,
                              success_rates, md) for c in vcols)
    if sr < 100:
        v += f" ({sr}%)" if md else rf" ({sr}\%)"
    return v


def _unit_header(p, vcols, md):
    """Units line for a p column: 's / MB / it/s' (it/s in 10^-3 for large p)."""
    its = "it/s" if p < ITS_SCALE_P else ("10⁻³ it/s" if md else r"$10^{-3}$ it/s")
    parts = [UNITS[c][1] or its for c in vcols]
    text = " / ".join(parts)
    return text if md else rf"{{\scriptsize\textit{{{text}}}}}"


def _panel(idx, fam, ns, ps, groups, marks, best, vcols, success_rates) -> str:
    """LaTeX tabular for one family: columns p, rows n x method; each cell holds
    the metrics in `vcols`. Set at \\footnotesize and only shrunk if too wide."""
    np_ = len(ps)
    ncol = 2 + np_
    def box(x):
        return rf"\makebox[{CELL_W}][c]{{{x}}}"
    pcols = " & ".join(box(rf"$p{{=}}{p}$") for p in ps)
    units = " & ".join(box(_unit_header(p, vcols, md=False)) for p in ps)
    lines = [
        r"\adjustbox{max width=\linewidth, max totalheight=0.36\textheight}{%",
        r"\footnotesize\setlength{\tabcolsep}{5pt}%",
        rf"\begin{{tabular}}{{l l {'c' * np_}}}",
        r"\toprule",
        rf"Basis & Method & {pcols} \\",
        rf" & & {units} \\",
        r"\midrule",
    ]
    for ni, n in enumerate(ns):
        if ni > 0:
            lines.append(r"\midrule")
        nlabel = f"{n:,}".replace(",", r"{,}")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{$n = {nlabel}$}}}}\\")
        for gi, (basis, methods) in enumerate(groups):
            lines.append(rf"\cmidrule(lr){{1-{ncol}}}" if gi > 0 else r"\midrule")
            for mi, m in enumerate(methods):
                cells = [box(_value(idx, fam, p, n, m, basis, vcols, marks, best,
                                    success_rates, md=False)) for p in ps]
                lines.append(" & ".join([basis if mi == 0 else "", _label(m)]
                                        + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines)


def _md_panel(idx, fam, ns, ps, groups, marks, best, vcols, success_rates) -> str:
    """Markdown version of _panel."""
    head = ["n", "Basis", "Method"] + [f"p={p} ({_unit_header(p, vcols, md=True)})"
                                        for p in ps]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join([":--"] * 3 + [":-:"] * (len(head) - 3)) + "|"]
    for n in ns:
        first_n = True
        for basis, methods in groups:
            for mi, m in enumerate(methods):
                cells = [_value(idx, fam, p, n, m, basis, vcols, marks, best,
                                success_rates, md=True) for p in ps]
                lines.append("| " + " | ".join([f"{n:,}" if first_n else "",
                                                basis if mi == 0 else "",
                                                _label(m, md=True)] + cells) + " |")
                first_n = False
    return "\n".join(lines)


UNITS = {"elapsed_s": ("time", "s"), "max_rss_mb": ("peak RSS", "MB"),
         "itr_per_sec": ("iterations per second", None)}


def main():
    raw = _load_bench()
    if raw is None:
        print("no benchmark data found — nothing to generate")
        return

    metric_cols = [c for c, *_ in METRICS if c in raw.columns]
    agg = (raw.groupby(["scenario", "method", "n"])[metric_cols]
              .median().reset_index())
    idx = agg.set_index(["scenario", "method", "n"]).sort_index()

    families, ns, ps, groups = _discover(agg)
    ps = [p for p in ps if p not in (1, 5)]   # omit p=1 and p=5 to keep the table compact
    metrics = [mt for mt in METRICS if mt[0] in metric_cols]
    print(f"  families={[f for f, _ in families]}  n={ns}  p={ps}")
    print(f"  basis groups={[(b, ms) for b, ms in groups]}")

    marks = _isolated_marks(idx, families, ns, ps, groups)

    # Bold = unique best median within the basis group.
    best_wx = _best_marks_unique(raw, families, ns, ps, groups, metrics)

    # Algorithmic success rate (%): completed / attempted reps per (fam, p, n, m).
    # Attempted = reps run so far, including reps recorded as failed after a
    # method was skipped for repeated timeouts; reps not yet run are excluded.
    success_rates = {}
    for fam, _ in families:
        for p in ps:
            for n in ns:
                for _, methods in groups:
                    for m in methods:
                        sc = f"{fam}_p{p:02d}"
                        t = raw.loc[
                            (raw["scenario"] == sc) &
                            (raw["method"] == m) &
                            (raw["n"] == n),
                            "elapsed_s"
                        ]
                        att, ok = len(t), int(t.notna().sum())
                        pct = round(100 * ok / att) if att else 0
                        if ok < att:
                            pct = min(pct, 99)   # never show 100 unless all completed
                        if ok > 0:
                            pct = max(pct, 1)    # never show 0 unless none completed
                        success_rates[(fam, p, n, m)] = (pct, att)

    blocks = [
        "% Auto-generated by public/make_scalability_table.py — do not edit by hand.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{amsmath}, "
        "\\usepackage{adjustbox}, and the \\Ricon / \\pyicon macros",
        "",
    ]
    vcols = [c for c, *_ in metrics]
    order = " / ".join(UNITS[c][0] + (f" ({UNITS[c][1]})" if UNITS[c][1] else "")
                       for c in vcols)
    bold = "; ".join(f"{'lowest' if d == 'min' else 'highest'} {UNITS[c][0]}"
                     for c, _, d, _ in metrics)
    caption = (
        r"Scalability of GAM fitting versus the number of predictors $p$ at each "
        r"sample size $n$, for (a) Gaussian (additive) and (b) logistic scenarios. Each "
        rf"cell gives medians over replicates as \texttt{{{order}}} (units in the "
        r"column headers; iterations per second in units of $10^{-3}$ for $p \ge "
        rf"{ITS_SCALE_P}$), followed by the algorithmic success rate in parentheses "
        r"when below 100\% (share of attempted replicates completed within the time "
        r"cap). Icons mark the programming environment (\Ricon{} R, "
        r"\pyicon{} Python). Methods are grouped by spline "
        r"basis. \textbf{Bold} = unique best median within basis group, per metric "
        rf"({bold}); ``" + CENSOR + r" (0\%)'' = no replicate completed within "
        r"the time cap (fitting abandoned); "
        r"$^\dagger$ marks a gapped completion (see text).")
    blocks += [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:scalability}",
        r"\par\medskip".join(
            rf"\textbf{{({'abcdefgh'[i]}) {flab}}}\par\smallskip" + "\n" +
            _panel(idx, fam, ns, ps, groups, marks, best_wx, vcols, success_rates)
            for i, (fam, flab) in enumerate(families)),
        r"\end{table}",
        "",
    ]
    md = ["# Scalability of GAM fitting", "",
          f"Each cell: `{order}` (medians over replicates), then the success rate in "
          "parentheses when below 100% (share of attempted replicates completed within "
          f"the time cap). For p ≥ {ITS_SCALE_P}, it/s is reported in units of 10⁻³. "
          "**Bold** = unique "
          f"best median within basis group, per metric ({bold}). [R] = R, [py] = Python. "
          "– (0%) = no replicate completed within "
          "the time cap (fitting abandoned); "
          "† = gapped completion.", ""]
    for i, (fam, flab) in enumerate(families):
        md += [f"## ({'abcdefgh'[i]}) {flab}", "",
               _md_panel(idx, fam, ns, ps, groups, marks, best_wx, vcols, success_rates),
               ""]

    out = TABLES_DIR / "table_scalability.tex"
    out.write_text("\n".join(blocks))
    print(f"saved {out}")
    out_md = TABLES_DIR / "table_scalability.md"
    out_md.write_text("\n".join(md))
    print(f"saved {out_md}")
    print(f"  {len(best_wx)} bold cells (unique best)")


if __name__ == "__main__":
    main()
