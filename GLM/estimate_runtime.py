#!/usr/bin/env python3
"""Extrapolate a pilot on the same machine; includes the untimed warm fits."""
import argparse,json
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument("--pilot",default="results/pilot");ap.add_argument("--config",default="configs/full.json");a=ap.parse_args()
cfg=json.loads(Path(a.config).read_text());df=pd.read_csv(Path(a.pilot)/"raw_results.csv")
# Each timing task has one warm fit + R repeats; remaining datasets have one warm + one measured fit.
k=len(cfg["timing_seeds"]);b=cfg["statistical_replicates"];fits=k*(cfg["timing_repeats"]+1)+(b-k)*2
df["estimated_fit_hours"]=pd.to_numeric(df.median_fit_seconds,errors="coerce")*fits/3600
df[["scenario","method","median_fit_seconds","estimated_fit_hours"]].to_csv(Path(a.pilot)/"estimated_full_runtime.csv",index=False)
print(df[["scenario","method","estimated_fit_hours"]].to_string(index=False))
print("Estimated fitting hours:",df.estimated_fit_hours.sum(),". Process startup, JIT compilation and data I/O are additional; pilot failures require inspection.")
