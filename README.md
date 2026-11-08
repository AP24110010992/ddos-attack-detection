# DDoS Attack Detection Using Machine Learning (CIC-DDoS2019)


A machine-learning system that classifies network-flow records as **BENIGN** or **DDoS ATTACK**, with an interactive Streamlit dashboard. Everything on the dashboard is read from files produced by the scripts in this repository, and no performance number is typed in by hand.

This is an **offline evaluation project**. It does not monitor live traffic, and a prediction is a model output, not proof of an attack.

## Results at a glance

Models were trained on one capture day (2018-12-01, "01-12") and tested **once** on a different day (2018-11-03, "03-11") that includes an attack type (NetBIOS) never seen in training.

| Model | Used for | Recall (attack) | False-positive rate | Balanced accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree, depth 4, 74 input columns (62 after constant columns are removed) | CSV upload page | 98.17% | 1.45% | 98.36% | 0.982 |
| Logistic Regression, 7 form-compatible features | Manual-entry page | 95.64% | 1.80% | 96.92% | 0.993 |

Read these numbers with the following in mind:

- **The scores are from one dataset recorded in one lab.** Performance on other networks is unknown.
- **The models were chosen after seeing the test-day results.** The selection rule was fixed beforehand, but the scores are slightly optimistic.
- **A random split would have overstated performance.** The Model Performance page shows the same models under three evaluation methods side by side.
- **Detection of the unseen attack type (NetBIOS) is unreliable.** It differs from 0% to 99.9% between models, so it is not evidence that any model detects unknown attacks.
- **Some test flows repeat training flows.** On test rows that are not exact copies of a training row, recall is 97.3% (tree) and 93.1% (logistic regression).
- **Precision and accuracy depend on the sample mix.** The working sample is 20% benign, while the raw data is 0.42% benign.

## Dataset

[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html), from the Canadian Institute for Cybersecurity, University of New Brunswick. The dataset's licence requires citing the dataset and this paper:

> Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak and Ali A. Ghorbani, "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.

Four of the official CSV files were obtained through a Kaggle mirror ("CIC-DDoS2019 30GB (Full Dataset CSV Files)"):

| Day | File | Rows |
|---|---|---|
| 01-12 (training) | `DrDoS_LDAP.csv` | 2,181,542 |
| 01-12 (training) | `Syn.csv` | 1,582,681 |
| 03-11 (testing) | `LDAP.csv` | 2,113,234 |
| 03-11 (testing) | `Syn.csv` | 4,320,541 |

That is 10,197,998 raw records, of which 42,918 (0.42%) are benign. A 20,000-row working sample (2,000 benign and 8,000 attack rows per day) is drawn from them by `src/build_sample.py`.

**The data is not included in this repository.** `data/raw/` and `data/processed/` are excluded by `.gitignore`.

## Method in brief

1. **Inspect** the raw files in chunks (`src/inspect_raw.py`): real column names, labels, missing and infinite values, duplicates.
2. **Sample** 20,000 rows while keeping the two capture days separate (`src/build_sample.py`).
3. **Check for leakage** (`src/leakage_check.py`). IP addresses, flow identifiers, timestamps, ports and the `Inbound` flag are excluded, because they describe this lab network and not traffic behaviour. Four exact duplicate columns are also removed.
4. **Preprocess** with one scikit-learn pipeline fitted on the training day only (`src/data_processing.py`): infinite values become NaN, NaN is filled with the training median, and constant columns are dropped.
5. **Compare** baselines (majority class, one-feature threshold), Decision Trees, a Random Forest and Logistic Regression with time-blocked validation inside the training day (`src/baselines.py`, `src/train_compare.py`).
6. **Evaluate once** on the unseen day (`src/final_evaluation.py`, `src/evaluation_report.py`), including a seen-versus-unseen attack-type breakdown and a comparison with a random split.
7. **Save** the two dashboard models and their metadata with Joblib (`src/train.py`).

## Project structure

```
ddos-attack-detection/
    app.py                     Streamlit dashboard (5 pages)
    requirements.txt           Pinned library versions
    README.md                  This file
    .gitignore                 Keeps data and the environment out of Git
    src/
        inspect_raw.py         Scan the raw CSV files
        build_sample.py        Draw the 20,000-row working sample
        describe_sample.py     Describe the sample
        leakage_check.py       Leakage and shortcut diagnostics
        eda.py                 Exploratory charts
        data_processing.py     Feature schema, validation, preprocessing pipeline
        compact_features.py    The 7-feature schema used by the manual form
        evaluate.py            Metrics and time-blocked validation
        baselines.py           Majority-class and one-feature baselines
        train_compare.py       Model comparison inside the training day
        compare_compact.py     Full versus compact feature sets
        compare_fill.py        Fill strategies for undefined rate values
        final_evaluation.py    Cross-day evaluation (opens the test day)
        evaluation_report.py   Duplicates, seen/unseen types, figures
        train.py               Train and save the two dashboard models
        predict.py             Shared prediction code
        manual_input.py        Manual-form logic and validation
        csv_prediction.py      CSV-upload logic and validation
        dataset_explorer.py    Dataset Explorer logic
        performance_data.py    Model Performance page logic
        dashboard_data.py      Helpers that read the artifact files
        make_demo_csv.py       Demo and deliberately broken CSV files
    models/                    Saved Joblib pipelines (small)
    artifacts/                 Evaluation results (JSON/CSV) and figures
    tests/                     64 automated tests (pytest)
    docs/                      Manual test log
    data/raw/                  Official CSV files (not in Git)
    data/processed/            Sample and demo files (not in Git)
    notebooks/                 Reserved for experiments
```

## Setup

Windows with Python 3.12 (developed on 3.12.3).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## Run the dashboard

```powershell
streamlit run app.py
```

It opens at http://localhost:8501. The saved models and result files are included in the repository, so the Overview, Manual Detection, CSV Prediction and Model Performance pages need no training. The Dataset Explorer's working-sample view needs the sample from `build_sample.py`.

| Page | What it does |
|---|---|
| Dashboard Overview | Dataset statistics, test-day metrics, confusion matrix, model comparison |
| Dataset Explorer | Preview, label distribution, missing values, statistics, downloads |
| Manual Detection | Enter one flow, press **DETECT**, see BENIGN or DDoS ATTACK with a probability |
| CSV Prediction | Upload flow records, get a prediction per row, download the results |
| Model Performance | Metrics with formulas, confusion matrix, evaluation-method comparison, feature importance |

The manual form uses duration in milliseconds. The two rate fields are calculated from the totals and the duration, and the model's training data stores duration in microseconds.

## Reproduce the whole pipeline

Put the four CSV files in `data\raw\01-12` (`DrDoS_LDAP.csv`, `Syn.csv`) and `data\raw\03-11` (`LDAP.csv`, `Syn.csv`), then run in this order:

```powershell
python src\inspect_raw.py
python src\build_sample.py
python src\describe_sample.py
python src\leakage_check.py
python src\eda.py
python src\baselines.py
python src\train_compare.py
python src\compare_compact.py
python src\compare_fill.py
python src\final_evaluation.py
python src\evaluation_report.py
python src\train.py
python src\make_demo_csv.py
```

Random seeds are fixed, and `train.py` refuses to save a model unless its test-day confusion counts equal those in `artifacts/final_evaluation.json`. To retrain after changing the data, rebuild the sample and run the last steps again.

## Tests

```powershell
python -m pytest tests -v
```

64 automated tests cover the preprocessing, metrics, manual-form validation, CSV handling, saved-model reproduction (including in a fresh Python process) and the consistency of the dashboard with the saved results. Some tests are skipped if the sample or artifact files have not been created. The manual checks are in `docs/manual_test_log.md`.

## Limitations

- One dataset, one test network, two capture days.
- Binary classification only (benign versus attack), not attack-type classification.
- Offline evaluation only. This is not a live monitoring tool.
- The working sample is rebalanced to 20% benign, so precision and accuracy do not reflect real traffic.
- The second most important feature of the tree, `URG Flag Count`, is set in about half of the benign flows and almost no attack flows in this data. Its mechanism is not explained, and it may reflect how the data was produced.
- Some attack flows are exact copies of others, which makes parts of the test easier.

## Future work (not implemented)

Training on both days with a new held-out set, multi-class attack-type detection, probability calibration, and evaluation on other datasets.