#!/usr/bin/env python3
"""Keep failure denominators; average independent datasets, not timing repeats."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS={"R_stats":"R: stats", "R_fastglm":"R: fastglm", "R_speedglm":"R: speedglm", "R_biglm":"R: biglm", "R_glmnet":"R: glmnet",
        "Py_statsmodels":"Python: statsmodels", "Py_sklearn":"Python: scikit-learn", "Py_glum":"Python: glum",
        "Py_spglm":"Python: spglm", "Py_glum_lasso":"Python: glum", "Py_skglm":"Python: skglm"}


def boolean(x):return x.astype(str).str.lower().isin(["true","1"])


def stats(values):
    x=pd.to_numeric(values,errors="coerce").dropna().to_numpy()
    mean=float(np.mean(x)) if len(x) else np.nan
    sd=float(np.std(x,ddof=1)) if len(x)>1 else np.nan
    return mean,sd,sd/np.sqrt(len(x)) if len(x)>1 else np.nan,len(x)


def tex_escape(x):
    return str(x).replace("\\",r"\textbackslash{}").replace("_",r"\_").replace("%",r"\%").replace("&",r"\&")


def cell(mean,sd):
    if not np.isfinite(mean):return "--"
    fmt=lambda v:f"{v:.3g}"
    return fmt(mean)+(r" $\pm$ "+fmt(sd) if np.isfinite(sd) else "")


def main_table(frame,path,penalized,profile):
    def pc(mm):return r">{\raggedright\arraybackslash}p{"+str(mm)+"mm}"
    widths=[15,30,23,23,20] if penalized else [20,35,27,27,24]
    cols=pc(widths[0])+pc(widths[1])+"rr"+pc(widths[2])+pc(widths[3])+"r"+pc(widths[4])
    if penalized:cols+=pc(19)
    caption=("Penalized Poisson GLMs" if penalized else "Unpenalized Poisson GLMs")
    if profile != "full":caption="VALIDATION ONLY ("+profile+"): "+caption
    headers=["Design","Implementation","n","p","Fit time (s)","Peak RSS (MiB)","Success","Test deviance"]
    if penalized:headers.append("F1")
    lines=[r"\begingroup\scriptsize\setlength{\tabcolsep}{2pt}",r"\begin{longtable}{"+cols+"}",r"\caption{"+caption+r". Means $\pm$ between-dataset SD; see notes.}\label{"+("tab:glm-pen" if penalized else "tab:glm-unpen")+r"}\\",
           r"\toprule"," & ".join(headers)+r" \\",r"\midrule\endfirsthead",r"\toprule"," & ".join(headers)+r" \\",r"\midrule\endhead",r"\bottomrule\endfoot"]
    for _,r in frame.iterrows():
        design=r["scenario"]+(f' ($\\delta={r["density"]:.2g}$)' if r["density"] < 1 else "")
        row=[design,LABELS[r["method"]],str(int(r["n"])),str(int(r["p"])),cell(r["fit_seconds_mean"],r["fit_seconds_sd"]),cell(r["peak_rss_mib_mean"],r["peak_rss_mib_sd"]),f'{int(r["successful_datasets"])}/{int(r["attempted_datasets"])}']
        row.append(cell(r["test_deviance_mean"],r["test_deviance_sd"]))
        if penalized:row.append(cell(r["f1_mean"],r["f1_sd"]))
        lines.append(" & ".join(row)+r" \\")
    lines += [r"\end{longtable}",r"{\footnotesize\noindent Timing is the mean of per-dataset median fitting times on the fixed timing seeds; RSS is the mean of sampled per-dataset peaks and includes the runtime and input matrices. Both use successful timing datasets. Success uses every attempted statistical dataset. Test deviance and F1 use successful statistical datasets; failed fits are not silently imputed. See the companion CSV for denominators, training deviance, MCSE, total fit time and throughput.\par}",r"\endgroup"]
    path.write_text("\n".join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args();out=Path(a.out)
    run=json.loads((out/"run.json").read_text());conf=run["config"]
    data=pd.read_csv(out/"raw_results.csv")
    data["success"]=boolean(data["success"]);data["timing_dataset"]=boolean(data["timing_dataset"])
    tables=out/"tables";tables.mkdir(exist_ok=True)
    groups=["scenario","regime","n","p","density","method"]
    summaries=[]
    metrics={"fit_seconds":"median_fit_seconds","peak_rss_mib":"peak_rss_mib", "train_deviance":"train_deviance",
             "test_deviance":"test_deviance","objective":"objective","f1":"f1","precision":"precision","recall":"recall","kkt_scaled":"kkt_scaled"}
    for key,group in data.groupby(groups,sort=False,dropna=False):
        row=dict(zip(groups,key));ok=group[group.success];tim=ok[ok.timing_dataset]
        row.update(attempted_datasets=len(group),successful_datasets=len(ok),success_rate=len(ok)/len(group),
                   success_mcse=np.sqrt((len(ok)/len(group))*(1-len(ok)/len(group))/len(group)),
                   attempted_timing_datasets=int(group.timing_dataset.sum()),successful_timing_datasets=len(tim),
                   requested_timing_fits=int(group.loc[group.timing_dataset,"requested_repeats"].sum()),
                   completed_timing_fits=int(group.loc[group.timing_dataset,"completed_repeats"].sum()),
                   successful_timing_fits=int(tim.completed_repeats.sum()),
                   total_fit_seconds=float(tim.total_fit_seconds.sum()),
                   fits_per_second=(tim.completed_repeats.sum()/tim.total_fit_seconds.sum() if tim.total_fit_seconds.sum()>0 else np.nan),
                   warning_datasets=int((group.warning_count>0).sum()),
                   boundary_selections=int(boolean(group.lambda_boundary).sum()))
        for label,column in metrics.items():
            source=tim if label in {"fit_seconds","peak_rss_mib"} else ok
            if label in {"f1","precision","recall"} and group.regime.iloc[0] not in {"highdim","sparse_highdim"}:
                source=source.iloc[:0]  # Unpenalized fits do not select support.
            m,sd,se,count=stats(source[column]);row.update({label+"_mean":m,label+"_sd":sd,label+"_mcse":se,label+"_n":count})
        summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(tables/"summary_all.csv",index=False)
    unpen=summary[~summary.regime.isin(["highdim","sparse_highdim"])]
    pen=summary[summary.regime.isin(["highdim","sparse_highdim"])]
    unpen.to_csv(tables/"table1_unpenalized.csv",index=False);pen.to_csv(tables/"table2_penalized.csv",index=False)
    main_table(unpen,tables/"table1_main.tex",False,conf["profile"])
    main_table(pen,tables/"table2_main.tex",True,conf["profile"])
    data[~data.success].to_csv(tables/"failures.csv",index=False)
    paired=[]
    for (scenario,seed),group in data[data.success & data.timing_dataset].groupby(["scenario","seed"],sort=False):
        base="R_glmnet" if group.regime.iloc[0] in ["highdim","sparse_highdim"] else "R_stats"
        ref=group[group.method.eq(base)]
        if len(ref)!=1:continue
        t0=float(ref.median_fit_seconds.iloc[0])
        for _,r in group.iterrows():
            if r.median_fit_seconds>0:paired.append({"scenario":scenario,"seed":seed,"method":r.method,"reference":base,"paired_speedup":t0/r.median_fit_seconds})
    paired=pd.DataFrame(paired,columns=["scenario","seed","method","reference","paired_speedup"])
    paired.to_csv(tables/"paired_speedups_raw.csv",index=False)
    speedups=[]
    for key,g in paired.groupby(["scenario","method","reference"],sort=False):
        m,sd,se,n=stats(g.paired_speedup)
        speedups.append(dict(zip(["scenario","method","reference"],key),mean=m,sd=sd,mcse=se,paired_datasets=n))
    pd.DataFrame(speedups).to_csv(tables/"paired_speedups.csv",index=False)
    # Numerical pairing checks use matched seeds and do not assert a universal ranking.
    audits=[]
    for key,g in data[data.success].groupby(["scenario","seed"],sort=False):
        penalized=g.regime.iloc[0] in ["highdim","sparse_highdim"]
        obj=g.objective.to_numpy();cv=g.cv_deviance.dropna().to_numpy()
        audits.append({"scenario":key[0],"seed":key[1],"successful_methods":len(g),
                       "objective_range":float(np.ptp(obj)) if len(obj)>1 else np.nan,
                       "selected_lambda_count":int(g.lambda_selected.nunique()) if penalized else 0,
                       "selected_cv_score_range":float(np.ptp(cv)) if len(cv)>1 else np.nan,
                       "max_scaled_kkt":float(g.kkt_scaled.max())})
    pd.DataFrame(audits).to_csv(tables/"paired_numerical_audit.csv",index=False)
    expected=sum((len(conf["penalized_methods"]) if s["regime"] in ["highdim","sparse_highdim"] else len(conf["unpenalized_methods"]))*conf["statistical_replicates"] for s in conf["scenarios"])
    unique=len(data.drop_duplicates(["scenario","seed","method"]))
    valid=bool(conf["profile"]=="full" and len(conf["scenarios"])==11 and unique==expected and len(data)==unique
               and len(conf["unpenalized_methods"])==8 and len(conf["penalized_methods"])==3)
    readiness={"profile":conf["profile"],"expected_dataset_method_tasks":expected,
               "completed_tasks":len(data),"successful_tasks":int(data.success.sum()),
               "structurally_complete_full_run":valid,
               "submission_gate":"Inspect failures, paired_numerical_audit.csv, CV boundary counts and environment before inserting tables. Structural completion alone does not certify statistical equivalence."}
    (out/"submission_readiness.json").write_text(json.dumps(readiness,indent=2))
    timing=sum(data.loc[data.timing_dataset,"completed_repeats"])
    tseeds=len(conf["timing_seeds"]);sreps=conf["statistical_replicates"]
    report=(f'The {conf["profile"]} run completed {len(data)} of {expected} dataset-method tasks; '
            f'{int(data.success.sum())} passed the recorded solver and independent optimality checks. '
            f'{int(timing)} timed fits were recorded on {tseeds} timing {"seed" if tseeds==1 else "seeds"}. '
            f'Statistical summaries use {sreps} independent {"dataset" if sreps==1 else "datasets"} per scenario, '
            'rather than treating repeated fits to one dataset as independent statistical replications.\n')
    if not valid:report="NOT FOR SUBMISSION: incomplete or validation profile.\n"+report
    (tables/"results_status.txt").write_text(report)
    # TeX metadata can be included in the revision files without hand-transcribing counts.
    (tables/"run_metadata.tex").write_text("\n".join([r"\newcommand{\GLMProfile}{"+tex_escape(conf["profile"])+"}",
         r"\newcommand{\GLMStatReplicates}{"+str(conf["statistical_replicates"])+"}",r"\newcommand{\GLMTimingSeeds}{"+str(len(conf["timing_seeds"]))+"}",
         r"\newcommand{\GLMTimingRepeats}{"+str(conf["timing_repeats"])+"}",r"\newcommand{\GLMTasks}{"+str(len(data))+"}",
         r"\newcommand{\GLMSuccessfulTasks}{"+str(int(data.success.sum()))+"}"]))
    print(report);print("Tables:",tables.resolve())


if __name__=="__main__":main()
