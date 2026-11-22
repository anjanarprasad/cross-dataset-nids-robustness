
# Cross-Dataset Network Intrusion Detection
### Robustness, Adaptation, and Deployment Efficiency

## About the Project

This project studies how machine learning models used for network
intrusion detection perform when they are tested on network traffic
from a different environment.

A model may perform well on the dataset it was trained on, but its
performance can decrease when it is tested on another dataset.
This can make it difficult to use the model in a real network.

The project also looks at ways to reduce model size and processing
time without losing too much detection performance.

This work was carried out as part of CSC 786: Computer Science
Problems at Dakota State University.

## Project Objectives

The main objectives are to:

1. Study how network intrusion detection models perform across
   different datasets.
2. Improve detection performance using a limited amount of
   labeled data from the target dataset.
3. Compare random sampling, standard reinforcement learning,
   and diversity-aware reinforcement learning for selecting
   target samples.
4. Compare INT8 quantization, structured pruning, and knowledge
   distillation for reducing model size.
5. Measure detection performance, model size, and inference time.

## Datasets

Two public network intrusion detection datasets were used:

- **NF-UNSW-NB15-v2:** Used for source model training.
- **NF-CSE-CIC-IDS2018-v2:** Used for target adaptation and evaluation.

The datasets are not included in this repository.

## What Was Implemented

The project includes:

- Data cleaning and feature preparation.
- Training a neural network to extract useful features.
- Selecting target samples using reinforcement learning.
- Training a classifier using a limited number of target labels.
- Evaluating the model on previously unused target records.
- Applying quantization, pruning, and knowledge distillation.
- Comparing model accuracy, storage size, and inference time.

## Main Findings

Diversity-aware sample selection improved multiclass attack
recognition compared with random sampling and standard
reinforcement learning.

The adapted model was also evaluated on 20,000 previously unused
records from the target dataset.

Among the compression methods tested, structured pruning gave
the best observed balance between detection performance and
model size. However, the improvement in inference speed was
relatively small.

The experiments also showed that some rare attack categories
were difficult to detect accurately.

## Repository Contents

This repository will include:

- Jupyter notebooks used for the experiments.
- Python scripts for the main implementation.
- Model evaluation and compression results.
- Figures showing the experimental findings.
- Information needed to reproduce the experiments.
- The final project report.

## Reproducibility

The experiments use fixed random seeds, documented data
preparation steps, and saved model configurations.

Instructions for setting up the environment and running the
experiments will be added as the final code is organized.

## Author

**Anjana Rajendra Prasad**

M.S. Computer Science (Artificial Intelligence)

Dakota State University
