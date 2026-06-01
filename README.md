# Lightweight Federated Intrusion Detection for Industrial IoT Using Pruned CNN-GRU Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.12](https://img.shields.io/badge/pytorch-2.12-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository contains the official implementation of the paper:

> **"Lightweight Federated Intrusion Detection for Industrial IoT Using Pruned CNN-GRU Networks"**

## Overview

A lightweight CNN-GRU architecture for Federated Learning-based intrusion detection in Industrial IoT networks. The model achieves **99.1% size reduction** (3.75 MB → 32 KB) through:

- **GRU** instead of BiLSTM (simpler gating, unidirectional)
- **Depthwise separable convolutions** for efficient feature extraction
- **Magnitude-based weight pruning** (one-shot, τ=0.02)
- **INT8 dynamic quantization** (post-training, no calibration data)

### Key results

| Dataset | Binary Acc | Multi-class Acc | Model Size | Inf. Time |
|---------|-----------|----------------|------------|-----------|
| X-IIoTID | 99.58% | — | 0.204 MB | 7.55 ms |
| WUSTL-IIoT-2021 | 100% | — | 0.204 MB | 6.43 ms |
| Edge-IIoTset | 100% | 95.47% (15 classes) | 0.204 MB | 3.84 ms |

## Repository Structure

```
├── models/                      # Model definitions
│   ├── baseline_cnn_bilstm.py          # Baseline CNN-BiLSTM
│   ├── lightweight_cnn_gru.py          # Proposed lightweight CNN-GRU
│   └── pruned_lightweight_cnn_gru.py   # Lightweight + weight pruning
├── utils/                       # Utilities
│   ├── data_loader.py            # Dataset loading and preprocessing
│   ├── metrics.py                # Evaluation metrics
│   └── synthetic_data.py         # Synthetic data generation
├── experiments/                 # Experiment scripts
│   ├── run_centralized.py        # Centralized training on real datasets
│   ├── run_federated.py          # Federated learning experiments
│   ├── run_multiclass.py         # Multi-class attack classification
│   ├── run_quantization.py       # INT8 quantization experiments
│   └── generate_figures.py       # Figure generation
├── data/                        # Dataset cache (downloaded at runtime)
├── results/                      # Experiment outputs and figures
├── paper/                        # Paper source (LaTeX + markdown)
│   ├── simple_paper.tex          # Article-class LaTeX source (Overleaf-ready)
│   ├── elsarticle.tex            # Elsevier elsarticle LaTeX source
│   ├── ieee_access.tex           # IEEE Access LaTeX source
│   ├── paper.md                  # Markdown source
│   ├── references.bib            # Bibliography
│   ├── highlights.md             # Journal highlights
│   └── *.png                     # Figures
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, scikit-learn, pandas

## Setup

```bash
# Clone the repository
git clone https://github.com/basitali08/iiot-ids-research
cd iiot-ids-research

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run all experiments

```bash
python -m experiments.run_all
```

### Individual experiments

```bash
# Centralized training on all datasets
python -m experiments.run_centralized

# Federated learning
python -m experiments.run_federated

# Multi-class classification
python -m experiments.run_multiclass

# INT8 quantization
python -m experiments.run_quantization

# Generate figures
python -m experiments.generate_figures
```

## Datasets

The following publicly available IIoT datasets are used:

- **X-IIoTID** — [IEEE Dataport](https://ieee-dataport.org/open-access/xiitoid-connectivity-and-device-agnostic-intrusion-dataset-industrial-internet-things)
- **WUSTL-IIoT-2021** — [Washington University](https://sites.wustl.edu/iiot-dataset/)
- **Edge-IIoTset** — [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)

Datasets are downloaded automatically to `data/` on first run.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ali2025lightweight,
  title={Lightweight Federated Intrusion Detection for Industrial IoT Using Pruned CNN-GRU Networks},
  author={Ali, Basit},
  year={2025},
  note={Preprint}
}
```

## License

This project is licensed under the MIT License.
