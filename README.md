# Cross-Dataset Network Intrusion Detection: Adaptation and Model Compression

This project evaluates how a machine-learning network intrusion detection system performs when a model trained on one traffic dataset is adapted to another. It also examines whether neural model compression can reduce storage and inference cost while maintaining attack detection performance.

A neural feature encoder is trained on **NF-UNSW-NB15-v2**. The target domain is **NF-CSE-CIC-IDS2018-v2**. Target-label selection is compared under a fixed annotation budget, followed by structured pruning, knowledge distillation, and dynamic INT8 quantization.

## Research objectives

- Examine the effect of cross-dataset distribution shift on network intrusion detection.
- Compare random selection, standard contextual-bandit reinforcement learning (RL), and diversity-aware RL for limited-label target adaptation.
- Assess the contribution of source-trained neural embeddings to target attack classification.
- Compare compact neural encoders using detection performance, model size, and CPU inference latency.
- Provide reproducible experiments, reusable Python code, and automated tests.

## Approach

The pipeline uses **43 flow-based features**: 37 retained numeric NetFlow features and six engineered flow statistics. A source-trained neural encoder produces a **32-dimensional embedding**, which is combined with the flow features to form a **75-dimensional input** to the adapted classifier.

For target adaptation, each selection method starts with the same 50 CIC records and acquires another 950 labels. A weighted logistic regression classifier is fitted on the 1,000 reviewed target records. Comparisons use three random seeds: **51, 73, and 97**.

The encoder is also compressed using:

- **Structured pruning:** 64 hidden units reduced to 32 or 16.
- **Knowledge distillation:** compact students trained using source labels and teacher outputs.
- **Dynamic INT8 quantization:** eligible teacher linear layers quantized for CPU inference.

All compression variants use the same 1,000 selected CIC labels for classifier adaptation and are compared on the CIC development partition.

## Main findings

| Experiment | Result | Evaluation scope |
|---|---|---|
| Diversity-aware RL | Mean macro-F1 **0.5857** across three seeds | CIC development |
| Standard RL | Mean binary PR-AUC **0.9539** across three seeds | CIC development |
| Frozen FP32 model | Macro-F1 **0.5460**; binary PR-AUC **0.9478** | 20,000-record CIC holdout |
| Pruned 16-unit encoder | Macro-F1 **0.5881**; binary PR-AUC **0.9586** | CIC development |
| Pruned 16-unit encoder | **43.3%** smaller serialized artifacts than FP32 teacher | Deployment artifact comparison |

Diversity-aware selection produced the strongest mean multiclass macro-F1, while standard RL achieved a slightly higher mean binary PR-AUC. The 16-unit pruned encoder gave the best measured compression-development results, with only a modest improvement in measured batch-256 CPU latency.

**Evaluation boundary:** The adaptation and compression comparisons use CIC development data. The saved 20,000-record holdout evaluation applies **only to the frozen FP32 classifier**. That holdout is separate from the adaptation and development records but comes from the same CIC traffic collection. These results do not establish zero-shot transfer or performance in a third independent network environment.

Detailed tables and limitations are in [Experimental Results](docs/results.md).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_source_model_and_adaptation.ipynb
│   ├── 03_evaluation_and_holdout.ipynb
│   ├── 04_model_compression.ipynb
│   └── 05_results_visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_encoder.py
│   ├── adaptation.py
│   ├── classification.py
│   ├── compression.py
│   ├── evaluation.py
│   └── inference.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_encoder.py
│   ├── test_adaptation.py
│   ├── test_compression.py
│   ├── test_evaluation.py
│   └── test_inference.py
└── docs/
    ├── methodology.md
    ├── reproducibility.md
    └── results.md
```

## Setup and execution

Python **3.11 or later** is recommended. Dependencies can be installed from the repository root:

```bash
python -m pip install -r requirements.txt
```

The notebooks are designed for **Google Colab** with input data and trained artifacts stored in Google Drive. The default dataset paths are:

```text
/content/drive/MyDrive/datanew/NF-UNSW-NB15-v2.csv
/content/drive/MyDrive/datanew/NF-CSE-CIC-IDS2018-v2.csv
```

The notebooks use `MyDrive/datanew/a4_analysis/` for sampled data, saved model artifacts, and experiment results. They are organized as follows:

| Notebook | Purpose |
|---|---|
| [01 — Data preparation](notebooks/01_data_preparation.ipynb) | Sample datasets and define the 43-feature schema |
| [02 — Source model and adaptation](notebooks/02_source_model_and_adaptation.ipynb) | Train/load the source encoder and compare target-label selection methods |
| [03 — Evaluation and holdout](notebooks/03_evaluation_and_holdout.ipynb) | Summarize development results and inspect saved frozen-holdout metrics |
| [04 — Model compression](notebooks/04_model_compression.ipynb) | Train and evaluate pruned, distilled, and INT8 variants |
| [05 — Results visualization](notebooks/05_results_visualization.ipynb) | Produce figures from saved evaluation results |

Notebook 01 prepares the cached data and feature schema. Notebook 02 creates the adapted-model outputs needed for notebooks 03 and 04. Notebook 05 uses results from the evaluation and compression stages.

The frozen-holdout reporting in notebooks 03 and 05 requires the previously saved `cell_44R_holdout_summary.json` and `cell_44R_category_results.csv`. **The provided notebooks do not independently regenerate the frozen-holdout evaluation.**

The complete execution order, artifact filenames, and experimental settings are described in the [Reproducibility Guide](docs/reproducibility.md).

## Automated tests

The `tests/` directory contains Pytest checks for data preparation, model shapes, target-label selection, compression, evaluation metrics, and the inference pipeline. The tests use synthetic records and temporary model checkpoints rather than the full traffic datasets.

```bash
python -m pytest tests/ -v
```

Passing the software tests checks implementation behavior; reproducing the reported dataset results additionally requires the corresponding data samples, fitted artifacts, and evaluation files.

## Batch inference

The inference module supports the adapted FP32 model and compressed model variants, provided their fitted artifacts are available:

```bash
python -m src.inference \
  --data-dir /path/to/a4_analysis \
  --input /path/to/network_flows.csv \
  --output /path/to/predictions.csv \
  --model teacher
```

It produces attack-category predictions and binary attack scores from network-flow inputs. See the [Reproducibility Guide](docs/reproducibility.md) for model choices and threshold handling.

## Documentation

- [Methodology](docs/methodology.md) — datasets, model architecture, adaptation, compression, and evaluation protocol.
- [Experimental Results](docs/results.md) — measured results, comparisons, confidence intervals, and limitations.
- [Reproducibility Guide](docs/reproducibility.md) — dependencies, execution order, paths, saved artifacts, and testing.

## Limitations

The experiments use sampled and imbalanced network-flow datasets, simulated label review from known ground-truth labels, and three acquisition seeds. Compression models were selected and evaluated on CIC development data. CPU timings depend on the Colab environment, and rare attack categories remain challenging. The independent holdout applies only to the frozen FP32 model and does not constitute evaluation on a new dataset or network environment.
