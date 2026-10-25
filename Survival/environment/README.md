# Environment

The completed run used R 4.5.1, Python 3.10.20 for non-neural models, and
Python 3.9.25 for DeepSurv. The two Python environments avoid incompatible
dependency requirements.

From the Survival directory:

```bash
conda env create -f environment/python_models.yml
conda env create -f environment/deepsurv.yml
Rscript environment/install_r_packages.R
```

The YAML files specify installation requirements. The *_versions.txt files list installed Python package versions. The original
pip freeze files also retain Conda build paths; use them as records, not as
portable installation files. R session information records the R libraries. The workers check the
model packages against code/config.json.

On the original cluster, some DeepSurv dependencies were installed in the
Python 3.9 user site. Do not disable that site when reusing the original
environment. A fresh environment may install the same versions inside the
environment itself. Use CPU execution as enforced by the run scripts.

Set SURV_PYTHON and DEEP_PYTHON to the full executable paths before submitting
jobs. The measured runtime and memory should be interpreted with the reported
hardware and environment.
