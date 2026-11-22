# Methodology

## 1. Research design

This project evaluates machine-learning-based network intrusion detection when a model trained on one traffic dataset is applied to another. The study has two main parts: adapting a source-trained model using a limited number of target labels, and reducing the model's deployment cost through compression.

The source dataset is **NF-UNSW-NB15-v2 (UNSW)** and the target dataset is **NF-CSE-CIC-IDS2018-v2 (CIC)**. A neural encoder is trained on UNSW. A separate classifier is then fitted using selected labeled CIC flows. Random sampling, standard contextual-bandit reinforcement learning (RL), and diversity-aware contextual-bandit RL are compared under the same label budget. Structured pruning, knowledge distillation, and dynamic INT8 quantization are evaluated as compression methods.

Because CIC labels are used to adapt the classifier, the principal target results measure **limited-label cross-dataset adaptation**, not zero-shot generalization.

## 2. Data preparation

### Datasets and sampling

The experiments use the NetFlow v2 versions of UNSW and CIC, which provide a common flow-feature schema and the `Label` and `Attack` fields. Full CSV files are read in chunks of 100,000 rows. Each row is sampled with a fixed probability chosen to produce approximately 100,000 records per dataset. The sampling seeds are 786 for UNSW and 787 for CIC.

Sampling is not stratified by attack type. The resulting class distributions therefore reflect the sampled records rather than a balanced design. Saved sample files are reused across subsequent experiment stages.

### Shared feature representation

The pipeline starts with **41 common numerical flow fields**, excluding IP addresses and the two label fields. Four fields are removed: `MIN_TTL`, `MAX_TTL`, `TCP_WIN_MAX_IN`, and `TCP_WIN_MAX_OUT`. This is a feature-design choice intended to reduce dependence on some network-specific measurements; it is not evidence that those fields cause distribution shift.

Six features are created from traffic volume, direction, packet count, and duration:

| Feature | Calculation | Description |
|---|---|---|
| `FE_LOG_TOTAL_BYTES` | `log1p(IN_BYTES + OUT_BYTES)` | Total exchanged bytes |
| `FE_LOG_TOTAL_PKTS` | `log1p(IN_PKTS + OUT_PKTS)` | Total packet count |
| `FE_BYTE_ASYMMETRY` | `(IN_BYTES - OUT_BYTES) / (total_bytes + 1)` | Directional byte imbalance |
| `FE_PKT_ASYMMETRY` | `(IN_PKTS - OUT_PKTS) / (total_pkts + 1)` | Directional packet imbalance |
| `FE_LOG_AVG_PKT_SIZE` | `log1p(total_bytes / (total_pkts + 1))` | Approximate packet size |
| `FE_LOG_DURATION` | `log1p(FLOW_DURATION_MILLISECONDS)` | Flow duration |

Nonnegative clipping is applied to the raw counts and duration before calculating these features. The final representation contains **37 retained raw features and 6 engineered features, or 43 features in total**. Their order is recorded in `csc786_feature_schema.json` and used consistently for training and inference.

### Data partitions and numerical transforms

Identical raw-feature records are grouped using feature hashes before splitting. UNSW is divided into training, validation, and test partitions, with nominal proportions of **70%, 15%, and 15%**, using seed 13. The recorded source training and validation sets contain **69,391** and **14,895** records.

CIC is separated into a target-label acquisition pool and a development evaluation partition using a nominal 50/50 group-aware split with seed 786. Duplicate feature signatures within development and any signatures overlapping the acquisition pool are removed from development. The recorded partitions contain **50,071 acquisition records** and **49,720 development records** after filtering.

Numerical preprocessing is fitted on **UNSW training records only**:

1. Apply `asinh` to `SRC_TO_DST_SECOND_BYTES` to reduce the influence of extreme values.
2. Fit a `StandardScaler` on the source training partition.
3. Apply the fitted scaler and then `asinh` to all 43 standardized values.

The same fitted transformation is applied to source validation and CIC records without refitting on target data. The resulting neural-network inputs use `float32` values.

## 3. Source-trained neural encoder

The source model predicts ten UNSW classes: Benign, Exploits, Fuzzers, Generic, Reconnaissance, DoS, Backdoor, Analysis, Shellcode, and Worms.

Its encoder has the architecture **43 → 64 → 32**, with layer normalization and GELU activations. The 32-dimensional output is used as a learned flow representation. A source-classification head maps this representation to ten logits. An auxiliary **32 → 64 → 43** decoder is used during denoising pretraining.

When training a new encoder, the procedure is:

1. Run **two epochs of denoising pretraining**. Randomly mask 15% of input values and minimize Smooth L1 reconstruction loss.
2. Train the encoder and classification head using a class-weighted focal loss. The focal exponent is **1.5**, and class weights are capped at **8**.
3. Use AdamW with learning rate **0.0007** and weight decay **0.0001** for supervised training, for up to **20 epochs**.
4. Select the checkpoint with the highest UNSW validation macro-F1, with early stopping after five epochs without improvement.

The workflow can instead load a saved source checkpoint when its input-feature and class ordering match the current schema. The **source-trained encoder is not fine-tuned with CIC labels** during the target-label selection comparison.

### Building the adapted classifier input

The source encoder produces 32 embedding values per flow. A `RobustScaler` with the 10th and 90th percentiles is fitted to `asinh`-transformed **UNSW training embeddings**. The same scaler is applied to target embeddings, with the transformed values clipped to **[-8, 8]**.

The 43 preprocessed flow features are concatenated with the 32 scaled embeddings. This creates a **75-feature input** for the adapted target classifier. The label-selection policy itself uses the 43 preprocessed flow features.

## 4. Target-label acquisition and adaptation

### Fixed annotation budget

The CIC acquisition pool contains records whose labels are initially treated as unavailable to the selection policy. Each method begins with the same **50 records**, selected using a fixed seed, and acquires **950 additional labels**, for a total budget of **1,000 reviewed records**. Ground-truth labels already present in the dataset simulate feedback after each selection; no human annotation was conducted.

The three selection methods are evaluated with seeds **51, 73, and 97**.

### Selection methods

**Random sampling** selects the remaining records uniformly without replacement after the shared initial sample.

**Standard RL** uses a contextual-bandit policy with the architecture **43 → 32 → 1**. At each step, it samples a candidate set of 256 unreviewed records, scores them, and chooses one using a mixture of softmax probabilities and **20% uniform exploration**. The selected record's label is revealed before updating the policy. An attack receives reward **+2**, while a benign record receives reward **−1**.

**Diversity-aware RL** uses the same policy and selection procedure. Its attack reward also depends on how often the discovered attack category has already been reviewed:

`reward = 2 + 6 / sqrt(1 + previous_category_count)`

Benign records still receive reward −1. The larger reward for a less frequently seen category encourages broader attack-category coverage. The policy is updated online with a reward-baseline policy-gradient loss and an entropy term. The two RL methods start from the same policy weights for each seed.

### Adapted classifier

For each selected set of 1,000 records, a multinomial-capable `LogisticRegression` classifier is fitted on the **75-feature CIC representations** using `lbfgs` and `max_iter=1200`.

Attack categories with fewer than three reviewed examples are combined into `Other_Attack` during classifier training. The remaining class labels are weighted by their reviewed frequency, with weights capped at **5**, to reduce dominance by frequent categories.

Every fitted classifier is evaluated on the same **CIC development partition**. These development labels are used to calculate results, not to fit the selection policy or the adapted classifier. Results are compared using equal label budgets, a common initial sample, and the three fixed seeds.

## 5. Model compression

Compression is applied to the source-trained neural encoder. The CIC adaptation labels are kept fixed at the **same 1,000 seed-51 diversity-aware selections** for all compression variants. Each pruned or distilled representation receives its own fitted embedding scaler and adapted classifier; the corresponding target labels remain the same.

### Structured pruning

The importance of each unit in the encoder's 64-unit hidden layer is estimated from the L2 norm of its outgoing weights. The **32** or **16** highest-scoring units are retained. Selected weights and layer-normalization parameters are copied to a compact encoder, which is then fine-tuned on **UNSW training records** for up to five epochs. UNSW validation macro-F1 selects the checkpoint.

The encoder parameter count changes from **5,024** for the teacher to **2,528** and **1,280** for the 32-unit and 16-unit variants, respectively. These counts refer only to the encoder, not to the complete deployment package.

### Knowledge distillation

Separate 32-unit and 16-unit student encoders are trained on UNSW data. Their objective combines three losses:

- **40% supervised focal loss** against UNSW class labels;
- **40% softened-logit KL divergence** between student and teacher, at temperature **2.0**;
- **20% Smooth L1 feature loss** between student and teacher's 32-dimensional embeddings.

Students train for up to eight epochs, with checkpoint selection based on UNSW validation macro-F1.

### Dynamic INT8 quantization

Dynamic INT8 quantization is applied to eligible linear layers of the FP32 teacher using PyTorch. Its embedding scaler and fitted CIC classifier remain unchanged, allowing the effect of quantization on the encoder to be assessed without training another target classifier.

### Compression evaluation and deployment measurements

All variants are compared on the same CIC development records. The reported deployment measurement includes preprocessing, neural encoding, and target classification, using CPU inference with **two PyTorch threads**. Batch sizes are **1, 32, and 256**. Model order is rotated during repeated timing runs, and both mean and 95th-percentile latency are recorded.

Serialized artifact size includes the encoder state, source feature scaler, embedding scaler, and adapted classifier. It excludes the Python runtime and installed dependencies. Device-specific CPU measurements are not assumed to generalize to other hardware.

## 6. Evaluation protocol

### Development metrics

For each adaptation or compression method, the main measures are:

- **Multiclass macro-F1:** the unweighted mean F1 across the actual CIC attack categories, excluding Benign from this average;
- **Exact attack-type accuracy:** the fraction of attack records assigned their exact attack category;
- **Binary PR-AUC / average precision:** computed from `1 − P(Benign)` using the binary CIC attack label;
- **Attack-category recall:** the fraction of records of each attack category correctly classified.

Binary ROC-AUC, threshold-based detection measures, and alert-budget precision and recall are also available in the evaluation code. A top-**12.5% alert budget** ranks records by their attack score and reports the precision and recall among the highest-scoring records; it is not equivalent to selecting a threshold from evaluation labels.

### Frozen CIC holdout

A separate saved evaluation covers **20,000 previously unused CIC records**, with **17,583 benign** and **2,417 attack** records. The seed-51 diversity-aware FP32 classifier was frozen before this evaluation. Known feature-signature overlaps were excluded. The model was not retrained on holdout labels.

This holdout is from the **same CIC traffic collection**, not from a third independent dataset. It therefore measures generalization to previously unused records from that collection. The repository's evaluation notebooks **load the saved holdout summary and category-level results**; they do not repeat the frozen-model test.

**Only the frozen FP32 model has this recorded holdout result.** The compressed models were compared on CIC development data, so their development metrics must not be presented as independent holdout results.

### Uncertainty estimates

For the development comparison of the teacher against the 16-unit pruned and distilled variants, **400 paired, attack-category-stratified bootstrap resamples** are used to estimate differences in macro-F1 and binary PR-AUC. The same resampled records are used for each model in a pair.

These intervals are conditional on the selected models and CIC development data. They do not account for model-selection bias or variation from training on different traffic collections. Similarly, three acquisition seeds are useful for observing run-to-run variability but do not establish statistical significance on their own.

## 7. Reproducibility and scope

The project records data-sampling and split seeds, acquisition seeds, feature ordering, selected target indices, model configurations, fitted scalers, and evaluation outputs. Reusable Python modules are provided in `src/`, and automated tests in `tests/` cover the computational pipeline using synthetic inputs. The executable notebooks organize the data preparation, source training and adaptation, evaluation, compression, and visualization stages.

The main limitations are class imbalance, sampling variation, simulated label review, development-set selection of compressed models, weak recognition of some rare attack categories, and CPU timing variability. The method does not demonstrate zero-shot generalization or deployment performance in a new network environment.

See [reproducibility.md](reproducibility.md) for execution details and [results.md](results.md) for the recorded findings.
