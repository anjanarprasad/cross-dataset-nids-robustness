# Experimental Results

## Study overview

This project studies network intrusion detection under cross-dataset distribution shift. A neural feature encoder was trained on **NF-UNSW-NB15-v2** and used to support classification on **NF-CSE-CIC-IDS2018-v2 (CIC)**. The experiments compare target-label selection strategies and smaller encoder architectures.

The source training and validation partitions contained **69,391** and **14,895** records. The sampled CIC data provided **50,071** records for target-label acquisition and **49,720** duplicate-filtered records for development evaluation. The model uses **43 flow features**, a **32-dimensional source-trained embedding**, and a **75-feature input** for the adapted classifier.

Source-to-target adaptation is **supervised**: the selection policy chooses CIC records and then receives their recorded ground-truth labels. Each run uses 50 common starting records and selects 950 more, for a fixed budget of **1,000 target labels**. The development results therefore measure limited-label adaptation, not zero-shot generalization.

## Target-label selection

Random selection, standard-reward contextual-bandit RL, and diversity-aware contextual-bandit RL were compared with identical label budgets at seeds 51, 73, and 97.

| Selection method | Seed 51 macro-F1 | Seed 73 macro-F1 | Seed 97 macro-F1 | Mean macro-F1 | Mean binary PR-AUC |
|---|---:|---:|---:|---:|---:|
| Random | 0.4986 | 0.3756 | 0.4179 | 0.4307 | 0.9463 |
| Standard-reward RL | 0.5137 | 0.5907 | 0.3973 | 0.5006 | **0.9539** |
| Diversity-aware RL | 0.5664 | 0.6025 | 0.5881 | **0.5857** | 0.9523 |

Diversity-aware RL improved mean macro-F1 by **0.0851** over standard-reward RL and **0.1550** over random selection. It also supported more attack types on average (9.33 compared with 8.33 for standard RL and 7.00 for random selection). Standard RL had a slightly higher mean binary PR-AUC, so the main observed advantage of diversity-aware selection is **multiclass attack-category recognition**, not binary ranking performance.

Three seeds show variability between runs but are too few to establish a reliable population-level significance claim.

### Contribution of neural features

With diversity-aware selection, the 43-feature classifier achieved **0.5612** mean development macro-F1, compared with **0.5857** after adding the 32 neural features. The mean difference was **+0.0245**. With standard-reward RL, the corresponding means were **0.4972** and **0.5006**. The benefit of the neural features varied with the selection method.

## Frozen CIC holdout

The seed-51 diversity-aware FP32 classifier was frozen before evaluation on **20,000 previously unused CIC records**, selected to avoid overlap with known feature signatures. This holdout remained part of the **same CIC data collection**, so it is not a test on a third independent network environment.

| Metric | Frozen FP32 holdout result |
|---|---:|
| Records | 20,000 |
| Benign / attack records | 17,583 / 2,417 |
| Multiclass macro-F1 | **0.545968** |
| Exact attack-type accuracy, among attacks | 0.917667 |
| Binary PR-AUC | **0.947813** |
| Binary ROC-AUC | 0.977146 |
| Precision at 12.5% alert budget | 0.905600 |
| Recall at 12.5% alert budget | 0.936698 |

At the 12.5% alert budget, the **2,500 highest-scoring flows** contained **2,264** of the **2,417** attacks. This is a ranking-based operating point, not a threshold selected using holdout labels.

Recognition varied considerably by category. **HOIC** recall was 0.9984 across 1,241 attack records, and **SSH-Bruteforce** recall was 1.0000 across 121 records. **Infilteration** recall was only 0.1691 across 136 records, while **Slowloris** recall was 0 across nine records. Categories with no holdout examples cannot support a recall conclusion.

## Compression: CIC development evaluation

Structured pruning reduced the encoder's hidden width from 64 units to 32 or 16. Knowledge distillation trained smaller source-domain student encoders. Dynamic INT8 quantization was applied to eligible teacher linear layers. Each compressed representation used the same selected **1,000 CIC labels** to fit its corresponding adapted classifier.

| Encoder variant | Development macro-F1 | Binary PR-AUC | Serialized artifacts (bytes) | Mean end-to-end batch-256 latency (ms) |
|---|---:|---:|---:|---:|
| FP32 teacher | 0.56636 | 0.94596 | 34,666 | 21.91 |
| Dynamic INT8 | 0.56777 | 0.94557 | 21,706 | 24.49 |
| Pruned-32 | 0.57341 | 0.94453 | 24,646 | 21.47 |
| **Pruned-16** | **0.58805** | **0.95864** | **19,654** | **20.06** |
| Distilled-32 | 0.58077 | 0.95083 | 24,682 | 22.27 |
| Distilled-16 | 0.58589 | 0.95122 | 19,690 | 20.37 |

**Pruned-16** achieved the strongest measured development macro-F1 and PR-AUC in this comparison. It reduced the encoder parameter count from **5,024 to 1,280** (**74.52% fewer parameters**) and combined serialized artifact size from **34,666 to 19,654 bytes** (**43.30% smaller**). These percentages refer to different quantities.

The measured batch-256 latency improvement over the FP32 teacher was modest: **21.91 ms to 20.06 ms**, or approximately **1.09×**. Dynamic INT8 reduced storage but was slower under the measured Colab CPU conditions. Artifact sizes exclude the Python runtime and dependencies, and observed timings depend on hardware and preprocessing costs.

### Development-set uncertainty

Four hundred paired, attack-category-stratified bootstrap resamples estimated the difference between each 16-unit encoder and the FP32 teacher on the same CIC development records.

| Comparison versus teacher | Macro-F1 difference | Approx. 95% interval | PR-AUC difference | Approx. 95% interval |
|---|---:|---:|---:|---:|
| Pruned-16 | +0.02169 | [0.01476, 0.02968] | +0.01268 | [0.00918, 0.01656] |
| Distilled-16 | +0.01952 | [0.01229, 0.02649] | +0.00526 | [0.00166, 0.00911] |

These intervals are **conditional on the selected models and CIC development partition**. They do not correct for model selection or variation across independently trained models and traffic collections. None of the compressed variants was evaluated on the frozen 20,000-record holdout.

## Interpretation and limitations

The results suggest that choosing target labels with attention to attack-category coverage can improve multiclass recognition under a limited annotation budget. A compact, source-trained neural encoder also produced promising development-set accuracy and storage trade-offs. However, performance remained weak for some rare attacks, especially Infilteration and Slowloris.

The main limitations are the three acquisition seeds, class imbalance, simulated label verification using existing dataset labels, development-based compression selection, and CPU timings measured in a shared Colab environment. The holdout supports a narrower finding of generalization to previously unused **CIC** records; it does not establish robustness to a new organization, traffic campaign, or unseen attack distribution.

## Numerical result files

The recorded experiment tables and holdout summary are stored in the `a4_analysis/` experiment directory:

| File | Contents |
|---|---|
| `cell_44K_three_method_comparison.csv` | Three-seed label-selection comparison |
| `cell_44M_encoder_summary.csv` | Neural feature ablation |
| `cell_44R_holdout_summary.json` | Frozen FP32 holdout metrics |
| `cell_44R_category_results.csv` | Holdout attack-category results |
| `cell_44AE_final_compression_comparison.csv` | Compression performance and deployment measurements |
| `cell_44AF_compression_confidence_intervals.csv` | Development-set paired bootstrap intervals |

The full setup, execution order, and evaluation boundaries are described in [`reproducibility.md`](reproducibility.md).
