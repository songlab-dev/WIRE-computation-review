"""One implementation, one dataset, one fresh process; evaluation is external."""
import argparse, json, os, random, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_config
from common import (
    versions,
    process_status,
    step_matrix,
    partial_loglik,
    breslow,
    survival_from_lp,
    lambda_max,
)

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    a = ap.parse_args()
    req = json.loads(Path(a.request).read_text())
    d = Path(req["data_dir"])
    out = Path(req["output_dir"])
    cfg = load_config()
    model = req["model"]
    seed = int(req["training_seed"])
    env = versions(
        cfg[
            "required_deep_versions"
            if model == "DeepSurv"
            else "required_python_versions"
        ]
    )
    read = lambda name: pd.read_csv(d / (name + ".csv"), float_precision="round_trip")
    train, val, test = read("train"), read("val"), read("test")
    scaling = read("scaling")
    features = [f"x{i+1}" for i in range(req["p"])]
    grid = np.loadtxt(d / "grid.csv")
    alphas = np.loadtxt(d / "penalties.csv") if model == "Coxnet" else None
    if model == "CoxPH":
        from lifelines import CoxPHFitter
    elif model == "Coxnet":
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    elif model == "RSF":
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    elif model == "Boosting":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.util import Surv
    elif model == "DeepSurv":
        import torch, torchtuples as tt
        from pycox.models import CoxPH

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    from threadpoolctl import threadpool_info, threadpool_limits

    threadpool_limits(limits=1)
    random.seed(seed)
    np.random.seed(seed)
    if model == "DeepSurv":
        torch.manual_seed(seed)
    before = process_status()
    details = {}
    (out / "fit_active").touch()
    start = time.perf_counter()
    x = train[features].to_numpy()
    xv = val[features].to_numpy()
    if model in ("Coxnet", "DeepSurv"):
        assert np.allclose(x.mean(axis=0), scaling["mean"], rtol=1e-12, atol=1e-12)
        assert np.allclose(
            x.std(axis=0, ddof=0), scaling["scale"], rtol=1e-12, atol=1e-12
        )
        x = (x - scaling["mean"].to_numpy()) / scaling["scale"].to_numpy()
        xv = (xv - scaling["mean"].to_numpy()) / scaling["scale"].to_numpy()
    if model == "CoxPH":
        fit = CoxPHFitter(baseline_estimation_method="breslow", penalizer=0.0)
        fit.fit(
            train[features + ["time", "event"]],
            duration_col="time",
            event_col="event",
            show_progress=False,
            fit_options={"max_steps": 500, "precision": 1e-7},
        )
        details["ties"] = "Efron"
        details["baseline"] = "Breslow"
    elif model == "Coxnet":
        maximum = lambda_max(x, train.time.to_numpy(), train.event.to_numpy())
        derived = np.geomspace(maximum, maximum * 0.01, 100)
        assert np.allclose(derived, alphas, rtol=1e-12, atol=1e-14)
        y = Surv.from_arrays(train.event.astype(bool), train.time)
        params = dict(
            l1_ratio=1.0,
            normalize=False,
            tol=cfg["coxnet"]["tolerance"],
            max_iter=cfg["coxnet"]["max_iterations"],
            fit_baseline_model=False,
        )
        fit = CoxnetSurvivalAnalysis(alphas=alphas, **params).fit(x, y)
        count = len(fit.alphas_)
        coefficients = [fit.coef_]
        assert np.allclose(fit.alphas_, alphas[:count], rtol=1e-12)
        # Native Coxnet can terminate an explicit path early. Complete omitted
        # penalties with native one-alpha fits; all of this is timed selection.
        for alpha in alphas[count:]:
            extra = CoxnetSurvivalAnalysis(alphas=[alpha], **params).fit(x, y)
            coefficients.append(extra.coef_)
        coef = np.concatenate(coefficients, axis=1)
        assert coef.shape == (len(features), len(alphas))
        scores = np.array(
            [
                partial_loglik(val.time.to_numpy(), val.event.to_numpy(), eta)
                for eta in (xv @ coef).T
            ]
        )
        best = int(np.argmax(scores))
        selected = coef[:, best]
        baseline = breslow(
            train.time.to_numpy(), train.event.to_numpy(), x @ selected, grid
        )
        details.update(
            selected_penalty=float(alphas[best]),
            selected_index=best,
            penalty_candidates=len(alphas),
            native_path_length=count,
            validation_partial_loglik=float(scores[best]),
        )
    elif model == "RSF":
        fit = RandomSurvivalForest(
            n_estimators=200,
            max_features=int(np.sqrt(req["p"])),
            min_samples_split=30,
            min_samples_leaf=1,
            max_depth=None,
            bootstrap=True,
            max_samples=len(train),
            n_jobs=1,
            random_state=seed,
            low_memory=False,
        ).fit(x, Surv.from_arrays(train.event.astype(bool), train.time))
        details.update(
            mtry=int(np.sqrt(req["p"])),
            trees=len(fit.estimators_),
            min_samples_split=30,
            min_samples_leaf=1,
            max_root_unique_samples=max(
                int(t.tree_.n_node_samples[0]) for t in fit.estimators_
            ),
            bootstrap_draws=len(train),
        )
    elif model == "Boosting":
        fit = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            max_leaf_nodes=4,
            min_samples_leaf=15,
            min_samples_split=30,
            max_features=None,
            subsample=1.0,
            dropout_rate=0.0,
            n_iter_no_change=None,
            random_state=seed,
        ).fit(x, Surv.from_arrays(train.event.astype(bool), train.time))
        baseline = breslow(
            train.time.to_numpy(), train.event.to_numpy(), fit.predict(x), grid
        )
        details.update(
            trees=int(fit.n_estimators_),
            max_leaves=max(int(t[0].tree_.n_leaves) for t in fit.estimators_),
        )
    elif model == "DeepSurv":
        x = x.astype("float32")
        xv = xv.astype("float32")
        y = (
            train.time.to_numpy().astype("float32"),
            train.event.to_numpy().astype("float32"),
        )
        yv = (
            val.time.to_numpy().astype("float32"),
            val.event.to_numpy().astype("float32"),
        )
        net = tt.practical.MLPVanilla(
            req["p"],
            [32],
            1,
            activation=torch.nn.ReLU,
            batch_norm=False,
            dropout=0.0,
            output_bias=False,
        )
        opt = torch.optim.Adam(
            net.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        fit = CoxPH(net, opt, device="cpu")
        cb = tt.callbacks.EarlyStopping(
            min_delta=0.0,
            patience=20,
            checkpoint_model=True,
            load_best=True,
            rm_file=True,
        )
        log = fit.fit(
            x,
            y,
            batch_size=cfg["deepsurv"]["batch_size"],
            epochs=150,
            callbacks=[cb],
            verbose=False,
            num_workers=0,
            shuffle=True,
            val_data=(xv, yv),
            val_batch_size=8224,
        )
        fit.compute_baseline_hazards()
        details.update(
            epochs=len(log.to_pandas()),
            training_rows=len(x),
            validation_rows=len(xv),
            batch_size=cfg["deepsurv"]["batch_size"],
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
        )
    elapsed = time.perf_counter() - start
    memory = process_status()
    (out / "fit_active").unlink()
    # Persist the fit boundary before any test prediction or audit conversion.
    record = {
        "fitting_time_s": elapsed,
        "peak_rss_kib": memory["VmHWM"],
        "peak_rss_mib": memory["VmHWM"] / 1024.0,
        "prefit_process": before,
        "postfit_process": memory,
        "environment": env,
        "details": details,
        "threadpools": threadpool_info(),
    }
    (out / "fit.json").write_text(json.dumps(record, indent=2))
    if model == "DeepSurv":
        history = log.to_pandas()
        history.to_csv(out / "training_log.csv", index=False)
        if not np.isfinite(history.to_numpy()).all():
            raise FloatingPointError(
                "Non-finite DeepSurv training/validation loss, even if best-checkpoint predictions are finite"
            )
    if model == "Coxnet":
        pd.DataFrame({"penalty": alphas, "validation_partial_loglik": scores}).to_csv(
            out / "penalty_selection.csv", index=False
        )
        np.savetxt(out / "selected_coefficients.csv", selected, fmt="%.17g")
    xt = test[features].to_numpy()
    if model in ("Coxnet", "DeepSurv"):
        xt = (xt - scaling["mean"].to_numpy()) / scaling["scale"].to_numpy()
    if model == "CoxPH":
        sf = fit.predict_survival_function(test[features])
        pred = step_matrix(sf.index.to_numpy(), sf.to_numpy().T, grid)
    elif model == "Coxnet":
        pred = survival_from_lp(xt @ selected, baseline)
    elif model == "RSF":
        pred = step_matrix(
            fit.unique_times_,
            fit.predict_survival_function(xt, return_array=True),
            grid,
        )
    elif model == "Boosting":
        pred = survival_from_lp(fit.predict(xt), baseline)
    else:
        sf = fit.predict_surv_df(xt.astype("float32"))
        pred = step_matrix(sf.index.to_numpy(), sf.to_numpy().T, grid)
        np.savez(
            out / "deep_training_audit.npz",
            x=x,
            xv=xv,
            time=y[0],
            event=y[1],
            val_time=yv[0],
            val_event=yv[1],
        )
        log.to_pandas().to_csv(out / "training_log.csv", index=False)
    np.savetxt(out / "survival.csv", pred, delimiter=",", fmt="%.17g")
    np.savetxt(out / "test_row_ids.csv", test.row_id.to_numpy(), fmt="%d")


if __name__ == "__main__":
    main()
