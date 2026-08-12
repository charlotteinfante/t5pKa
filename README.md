# T5pKa: a sequence-based model for microstate and pKa prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version: 2.0.0rc1](https://img.shields.io/badge/version-2.0.0rc1-blue.svg)](#versioning)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/charlotteinfante/t5pka_checkpoint/tree/main)
[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-Dataset-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20089806)

> **Release candidate with updated dependencies:** The `tweaks` branch contains
> T5pKa 2.0.0rc1, modernized for Python 3.11–3.12 and current package versions.
> For the original Python 3.9 and Torch 1.7.1 environment, use the
> [`main` branch](https://github.com/charlotteinfante/t5pKa/tree/main).

Predictions of pKa values provide insight into key aspects of molecular behavior,
including solubility, lipophilicity, and binding affinity. Despite their
importance, experimental microscopic pKa data remain scarce, creating a
bottleneck in training accurate prediction models. In addition, inconsistent
terminology across commonly used datasets hinders effective model development
and benchmarking.

While recent advances have been driven largely by graph-based neural networks,
the potential of sequence-based deep learning for pKa prediction remains
underexplored. T5Chem, a sequence-based multitask chemical reaction model,
offers an attractive way to cast molecular protonation/deprotonation as a
language-modeling task and couple microstate generation with subsequent pKa
estimation.

To pursue this direction, we introduce pKaCHU (pKa data that are Combined,
Honed, and Updated), a curated dataset comprising approximately 9,000
experimentally derived microscopic pKa entries with ionization-state
annotations. We also present T5pKa, a text-based transformer model for
small-molecule pKa prediction built on T5Chem.

T5pKa uses multitask learning to enumerate microstates, enabling both
protonation and deprotonation to be predicted by a single sequence-to-sequence
model. A separate regression model then predicts microscopic pKa values from
the resulting microstate pairs.

## Installation

### Requirements

- Python 3.11 or 3.12
- A recent version of `pip`
- Git, when installing directly from GitHub

### Install from the `tweaks` branch

```bash
git clone --branch tweaks --single-branch https://github.com/charlotteinfante/t5pKa.git
cd t5pKa

conda create --name t5pKa python=3.12
conda activate t5pKa

python -m pip install --upgrade pip
python -m pip install .
```

Alternatively, install the branch directly:

```bash
python -m pip install "t5pKa @ git+https://github.com/charlotteinfante/t5pKa.git@tweaks"
```

For a reproducible installation, replace `tweaks` with a release tag or full
commit hash.

Optional dependencies are available for analysis and development:

```bash
python -m pip install ".[analysis]"
python -m pip install ".[dev]"
```

The package requires Torch 2.12 or later. For a CUDA-specific Torch build,
install the appropriate build using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing T5pKa.

Confirm the installed version:

```bash
t5pka --version
```

## Demo

The T5pKa demo is available on
[Hugging Face Spaces](https://huggingface.co/spaces/charlotteinfante/t5pka-demo).

## Datasets

T5pKa uses two datasets:

1. Instructions for obtaining the calculated pKa data are available in the
   [t5pKa-data repository](https://github.com/charlotteinfante/t5pKa-data).
2. The experimental pKa dataset is available on
   [Zenodo](https://zenodo.org/records/20089807).

The calculated dataset cannot be redistributed directly because of its
licensing restrictions.

## Model checkpoints

Pretrained model checkpoints are available on
[Hugging Face](https://huggingface.co/charlotteinfante/t5pka_checkpoint/).

## Run predictions

After installation, use the `t5pka predict` command. T5pKa automatically uses
a CUDA device when one is available and otherwise runs on the CPU.

### Single-molecule predictions

Predict a protonated microstate:

```bash
t5pka predict \
  --smiles "Prot:Brc1ccc(C2CN3C=CSC3=N2)cc1" \
  --model_dir /path/to/sequence-to-sequence-checkpoint \
  --prediction protonated_prediction.csv
```

Predict a deprotonated microstate:

```bash
t5pka predict \
  --smiles "Deprot:C[C@@H](O)C(=O)O" \
  --model_dir /path/to/sequence-to-sequence-checkpoint \
  --prediction deprotonated_prediction.csv
```

Predict a microscopic pKa value:

```bash
t5pka predict \
  --smiles "C[C@@H](O)C(=O)O>>C[C@@H](O)C(=O)[O-]" \
  --model_dir /path/to/regression-checkpoint \
  --scaler /path/to/MinMaxScaler.gz \
  --prediction pka_prediction.csv
```

### Bulk predictions

`--data_dir` may point to a directory containing `test.source` or directly
to a named `.source` file:

```bash
t5pka predict \
  --data_dir /path/to/test-data \
  --model_dir /path/to/model-checkpoint \
  --prediction predictions.csv
```

Run the regression ensemble:

```bash
python -m t5pka.ensemble_prediction \
  --data_dir /path/to/test-data \
  --model_dir /path/to/ensemble-models \
  --scaler_random /path/to/random-split-scaler \
  --scaler_scaffold /path/to/scaffold-split-scaler \
  --prediction ensemble_predictions.csv
```

## Training

Train a regression model:

```bash
t5pka train \
  --data_dir /path/to/train-folder \
  --output_dir /path/to/output-directory \
  --task_type micropka \
  --pretrain /path/to/pretrained-model \
  --num_epoch 150 \
  --batch_size 128 \
  --init_lr 5e-4
```

Train a sequence-to-sequence model:

```bash
t5pka train \
  --data_dir /path/to/train-folder \
  --output_dir /path/to/output-directory \
  --task_type mixed \
  --pretrain /path/to/pretrained-model \
  --num_epoch 150 \
  --batch_size 128 \
  --init_lr 6e-4
```

Run pretraining:

```bash
t5pka pretrain \
  --data_dir /path/to/pretraining-data \
  --output_dir /path/to/output-directory \
  --tokenizer simple
```

## Versioning

- `1.0.0`: original reproducible environment on `main` (Python 3.9 and
  Torch 1.7.1)
- `2.0.0rc1`: release candidate on `tweaks` with updated Python and package
  support

The release-candidate suffix will be removed after the updated environment and
model workflows have been validated.

## License

T5pKa is distributed under the [MIT License](LICENSE).
