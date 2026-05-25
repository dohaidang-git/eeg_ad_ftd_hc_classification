# MLflow Pipeline

This project now includes a paper-driven MLflow runner for EEG classification using the `Cleaned_Epochs` dataset.

Implemented method:
- input data: band-specific 5-second EEG epochs from `Cleaned_Epochs`
- connectivity metrics: `cov`, `corr`, `xcov`, `xcorr`, `csd`, `coh`, `mi`, `ecc`, `aecov`, `aecorr`, `plv`, `wplv`
- base classifier: `FgMDM` from `pyriemann`
- meta-classifier: Elastic Net logistic regression
- evaluation: leave-one-subject-out cross-validation with subject-level probability aggregation
- tracked tasks: `AD vs HC`, `FTD vs HC`, `FTD vs AD`

Feature sources:
- `from_epochs`: compute connectivity from `Cleaned_Epochs`
- `from_precomputed`: load ready-made domain features from `Final_MultiDomain_Features_Role3(1)`

Run a smoke test first:

```bash
cd /home/dohaidang/DataMining_Project
python scripts/run_mlflow_experiment.py \
  --problem ad_hc \
  --bands alpha \
  --metrics cov,corr \
  --inner-folds 3 \
  --outer-limit 4
```

Run the full paper-style search space:

```bash
cd /home/dohaidang/DataMining_Project
python scripts/run_mlflow_experiment.py --problem all
```

Run the fast branch from precomputed domain features:

```bash
cd /home/dohaidang/DataMining_Project
python scripts/run_mlflow_experiment.py \
  --feature-source from_precomputed \
  --problem all \
  --bands alpha,theta,beta \
  --metrics cov,corr,plv
```

Open MLflow UI locally:

```bash
cd /home/dohaidang/DataMining_Project
mlflow ui --backend-store-uri ./mlruns
```

Notes:
- the full run is computationally heavy because it computes connectivity matrices for many band-metric combinations and applies nested cross-validation
- computed connectivity matrices are cached under `.cache/connectivity`
- the paper itself is binary per problem, so this runner logs three binary experiments rather than a single 3-class classifier
