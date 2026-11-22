# Reproducibility

This project evaluates how a network intrusion detection model trained on NF-UNSW-NB15-v2 performs when adapted to NF-CSE-CIC-IDS2018-v2. It compares three ways of selecting target labels and measures the effect of model compression on detection quality and CPU inference cost.

The main workflow runs in Google Colab with datasets and fitted artifacts stored in Google Drive. The `src/` modules provide reusable implementations, and `tests/` checks their behavior using synthetic data.

## Environment

Python 3.11 or later is recommended. Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The notebooks use NumPy, pandas, scikit-learn, PyTorch, Matplotlib, joblib, and IPython. GPU acceleration is optional for source-model training. The compression benchmarks use CPU inference with two PyTorch threads; latency depends on the machine and software environment. The dependency ranges permit different package versions, so an exact environment rebuild also requires recording the resolved versions.

## Data and feature preparation

The source dataset is **NF-UNSW-NB15-v2** and the target dataset is **NF-CSE-CIC-IDS2018-v2**. Both must use the NetFlow v2 column schema, including `Label` and `Attack`.

In the default Colab configuration, the input files are located at:

```text
/content/drive/MyDrive/datanew/
├── NF-UNSW-NB15-v2.csv
├── NF-CSE-CIC-IDS2018-v2.csv
└── a4_analysis/
    ├── UNSW_100k.csv
    ├── CIC_100k.csv
    └── csc786_feature_schema.json
```

`01_data_preparation.ipynb` reuses the two sample CSVs if they already exist. Otherwise, it reads each full CSV in 100,000-row chunks and performs Bernoulli sampling with a target of approximately 100,000 rows per dataset. The sampling seeds are 786 for UNSW and 787 for CIC. These are approximate sample sizes, not class-balanced samples. Reusing the same cached sample files is important for comparing repeated runs.

The feature schema begins with 41 common numerical fields. It excludes IP addresses and class labels, removes `MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_IN`, and `TCP_WIN_MAX_OUT`, and adds six flow features. This produces **43 ordered model features**. The exact feature order is stored in `csc786_feature_schema.json`.

The numerical preprocessing includes an inverse hyperbolic sine transformation on `SRC_TO_DST_SECOND_BYTES`, a standard scaler fitted on the UNSW training partition, and an inverse hyperbolic sine transformation after scaling. A source-fitted embedding scaler is applied to the encoder's 32-dimensional output. The adapted classifier receives **75 inputs: 43 flow features and 32 neural features**. Target labels are not used to fit the source scaler or embedding scaler.

## Execution order

Run the notebooks in numerical order from the `notebooks/` directory. Each notebook can be opened in Colab and run from the first cell. Paths at the beginning of the notebooks can be changed for another environment.

| Notebook | Main operation | Saved output folder |
|---|---|---|
| `01_data_preparation.ipynb` | Create/load dataset samples and save the feature schema | `a4_analysis/` |
| `02_source_model_and_adaptation.ipynb` | Train/load the UNSW encoder; compare target-label selection methods; fit the adapted CIC classifier | `a4_analysis/reproduction_runs/source_adaptation/` |
| `03_evaluation_and_holdout.ipynb` | Summarize three-seed development results and check the saved frozen-holdout results | `a4_analysis/reproduction_runs/evaluation_results/` |
| `04_model_compression.ipynb` | Prune and distill the encoder, apply dynamic INT8 quantization, and compare CIC development performance and CPU latency | `a4_analysis/reproduction_runs/model_compression/` |
| `05_results_visualization.ipynb` | Generate figures from the saved numerical results | `a4_analysis/reproduction_runs/visualizations/` |

Notebook 02 loads `a4_analysis/cell_44C_multiclass_source.pt` if that checkpoint is present and the saved schema matches. If it is absent, the notebook trains a new source model and saves `source_encoder_checkpoint.pt` in `reproduction_runs/source_adaptation/`. The chosen checkpoint is recorded in `run_config.json` and used by the compression workflow.

### Holdout result inputs

Notebook 03 requires the following fixed evaluation files in `a4_analysis/`:

```text
cell_44R_holdout_summary.json
cell_44R_category_results.csv
```

It optionally checks `cell_44R_holdout_predictions.csv` when record-level predictions are available. Notebook 05 also requires the holdout summary and category table to plot frozen-holdout recall. **The holdout files are not created by notebooks 01–04.** Without them, the holdout reporting cells cannot run. The saved holdout results refer to a previously frozen classifier; they must not be described as the independent holdout performance of a newly retrained model unless that specific model was evaluated and recorded separately.

## Experimental settings

The source data is separated with group-aware splits to limit duplicate-record leakage. The target sample is split into a label-acquisition pool and a CIC development set. Development records are filtered for duplicate feature signatures and overlap with the acquisition pool. The source split uses seed 13; the target split uses seed 786.

The label-selection experiments use a **1,000-record target review budget**, with the same 50 starting records for every method. Random selection, standard contextual-bandit RL, and diversity-aware RL are compared with seeds **51, 73, and 97**. The saved adapted classifier uses the diversity-aware selection from seed 51. The target classifier is a weighted logistic regression trained on reviewed CIC labels.

Compression includes structured-pruned encoders with hidden widths of 32 and 16, distilled students with the same widths, and dynamic INT8 quantization of the teacher's linear layers. For the compressed-model comparison, the same 1,000 reviewed target records are used to fit the corresponding classifiers. Measurements include multiclass macro-F1, binary PR-AUC, model storage, and CPU latency at batch sizes 1, 32, and 256. These compression results are **CIC development-set measurements**, not evaluations on the frozen holdout.

The frozen holdout consists of **20,000 distinct CIC records**, including **2,417 attack records**. It is separate from CIC adaptation and development, but still comes from the same CIC dataset and traffic campaign. It is therefore not evidence of generalization to a third, independent network environment.

## Saved results

The main result files are:

```text
a4_analysis/reproduction_runs/
├── source_adaptation/
│   ├── run_config.json
│   ├── adaptation_three_seed_results.csv
│   ├── source_standard_scaler.joblib
│   ├── source_embedding_scaler.joblib
│   ├── adapted_classifier_seed51.joblib
│   └── selected_indices_seed51.npy
├── evaluation_results/
│   ├── adaptation_three_seed_means.csv
│   ├── frozen_holdout_key_results.csv
│   └── frozen_holdout_category_results.csv
├── model_compression/
│   ├── compression_settings.json
│   ├── cic_development_comparison.csv
│   ├── cic_development_category_recall.csv
│   ├── cpu_inference_latency.csv
│   ├── model_storage.csv
│   └── final_compression_comparison.csv
└── visualizations/
    └── Fig*.png, Fig*.pdf, Fig*.svg
```

Model checkpoints and variant-specific scaler/classifier files are saved alongside the compression tables. A fresh run can produce different metrics if its sampled records, checkpoint, package versions, or compute environment differ. Source training and target adaptation are stochastic even when seeds are fixed.

## Software tests

From the repository root, run:

```bash
python -m pytest tests/ -v
```

The tests cover preprocessing, neural features, target-label selection, compression, evaluation, and the inference pipeline. They use synthetic inputs and temporary checkpoints. Passing these tests confirms specific implementation behavior; it does **not** establish that the recorded dataset metrics have been reproduced.

## Running inference

Once the required schema, fitted scalers, checkpoints, and adapted classifiers are available in the data directory, batch predictions can be generated with:

```bash
python -m src.inference \
  --data-dir /path/to/a4_analysis \
  --input /path/to/network_flows.csv \
  --output /path/to/predictions.csv \
  --model teacher
```

Valid model choices are `teacher`, `teacher_fp32`, `teacher_int8`, `pruned_32`, `pruned_16`, `distilled_32`, and `distilled_16`. The `teacher` option uses the adapted classifier saved by notebook 02. The `teacher_fp32` and `teacher_int8` options use the classifier and embedding scaler fitted during the compression comparison. The pruned and distilled variants use their own fitted artifacts. An explicit `--threshold` can be supplied for binary alert decisions; no threshold is selected from inference labels.

## Evaluation boundaries

- **Source validation:** UNSW data is used for source-model training and checkpoint selection.
- **CIC development:** CIC labels are reviewed for adaptation; the remaining development partition measures selected models and compression methods. These results can be affected by model or method selection.
- **Frozen CIC holdout:** The saved fixed-model evaluation is reported separately. The current notebooks read and verify its result files; they do not run a new holdout evaluation.
- **Reproducibility:** Fixed seeds, feature ordering, saved configuration, explicit artifact paths, and automated tests support rerunning the workflow. Exact numerical agreement additionally depends on using the same sampled records, trained checkpoints, and environment.
