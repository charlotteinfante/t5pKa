# T5pKa: a sequence-based model for microstate and pKa prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face%20Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/charlotteinfante/t5pka_checkpoint/tree/main)
[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-Dataset-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.18704856) 

  Predictions of pKa values provide insight into key aspects of molecular behavior, including solubility, lipophilicity, and binding affinity. Despite their importance, experimental microscopic pKa data remain scarce, creating a bottleneck in training accurate prediction models. In addition, inconsistent terminology across commonly used datasets hinders effective model development and benchmarking. While recent advances have been driven largely by graph-based neural networks, the potential of sequence-based deep learning for pKa prediction remains underexplored. T5Chem, a sequence-based multitask chemical reaction model, offers an attractive way to cast molecular protonation/deprotonation as a language modeling task, and to couple microstate generation with subsequent pKa estimation. To pursue this direction, we introduce pKaCHU (pKa data that are Combined, Honed, and Updated), a curated dataset comprising 9,042 experimentally derived microscopic pKa entries with ionization-state annotations. We also present T5pKa, a text-based transformer model for small molecule pKa prediction built on T5Chem. T5pKa leverages multitask learning to enumerate microstates, enabling both protonation and deprotonation to be predicted by a single sequence-to-sequence model, and then predicts microscopic pKa values from the resulting microstate pairs using a separate regression model. Across benchmark datasets, T5pKa achieves performance comparable to established pKa prediction tools and published models, while offering the advantage of a unified multitasking framework for microstate enumeration and microscopic pKa prediction. 



## Installation

### Clone the repo
```bash
git clone https://github.com/charlotteinfante/t5pKa.git
cd t5pKa
```

### Create the environment 
```bash
conda create -n t5pKa python=3.9
conda activate t5pKa
pip install .
```

### Install Compatiable PyTorch
This repo requires Torch 1.7.1 (CUDA 11.0 build)
```bash
pip uninstall -y torch
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Demo
T5pKa demo can be seen on Hugging Face Spaces: https://huggingface.co/spaces/charlotteinfante/t5pka-demo

## Datasets
There are two datasets that are used in this model: (1) the calculated pKa data and (2) the experimental pKa dataset. Due to licensing restrictions, we set up a repository with instructions on how to get the calculated pKa data found in https://github.com/charlotteinfante/t5pKa-data. The experimental pKa data can be found on [Zenodo](https://zenodo.org/records/18704856). 

## Run Predictions
Find model checkpoints [here](https://huggingface.co/charlotteinfante/t5pka_checkpoint/) !

Run single prediction:

**(A)** Predict protonated area of molecule
```bash
python run_prediction.py --smiles "Prot:Brc1ccc(C2CN3C=CSC3=N2)cc1" --model_dir ~/T5pKa_Review/SEQUENCETOSEQUENCE/ --prediction ~/T5pKa_Review/SEQUENCETOSEQUENCE/prediction.csv
```

**(B)** Predict deprotonated area of molecule
```bash
python run_prediction.py --smiles "Deprot:C[C@@H](O)C(=O)O" --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv
```

**(C)** Predict pKa of molecule

```bash
python run_prediction.py --smiles "Brc1ccc(C2C[NH+]3C=CSC3=N2)cc1>>Brc1ccc(C2CN3C=CSC3=N2)cc1" --model_dir ~/T5pKa_Review/REGRESSION/ --prediction ~/T5pKa_Review/REGRESSION/prediction.csv --scaler ~/T5pKa_Review/REGRESSION/
```
```bash
python run_prediction.py --smiles "C[C@@H](O)C(=O)O>>C[C@@H](O)C(=O)[O-]" --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv --scaler /path/to/scaler
```


Run bulk prediction:

**(A)** Predict the ionization of more than one molecule
```bash
python run_prediction.py --data_dir ~/testsets/NOVARTIS/MICROSTATE/ --model_dir ~/T5pKa_Review/SEQUENCETOSEQUENCE/ --prediction ~/T5pKa_Review/SEQUENCETOSEQUENCE/novartis_prediction.csv
```

**(B)** Predict the pKa of more than one molecule
```bash
python run_prediction.py --data_dir /path/to/data --model_dir ~/T5pKa_Review/REGRESSION/ --prediction ~/T5pKa_Review/REGRESSION/prediction.csv --scaler ~/T5pKa_Review/REGRESSION/
```

**(C)** Predict the pKa of more than one molecule using ensemble model
```bash
python ensemble_prediction.py --data_dir ~/testsets/NOVARTIS/MICROPKA/ --model_dir ~/REGRESSION/ --prediction ~/REGRESSION/nov_ensemble_prediction.csv --scaler_random ~/REGRESSION/RANDOM/ --scaler_scaffold ~/REGRESSION/SCAFFOLD/
```

## Training 
To train regression model
```bash
python __main__.py train --data_dir /path/to/train_folder/ --output_dir /path/to/output_directory/ --task_type micropka --pretrain /path/to/pretrained_model/ --num_epoch 150 --batch_size 128 --init_lr 5e-4
```

To train sequence-to-sequence model
```bash
python __main__.py train --data_dir /path/to/train_folder/ --output_dir /path/to/output_directory/ --task_type mixed --pretrain /path/to/pretrained_model/ --num_epoch 150 --batch_size 128 --init_lr 6e-4
```