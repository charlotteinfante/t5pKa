# T5pKa: a sequence-based model for microstate and pKₐ prediction
  Predictions of p$K_a$ values provide insight into key aspects of molecular behavior, including solubility, lipophilicity, and binding affinity. Despite their importance, experimental microscopic p$K_a$ data remain scarce, creating a bottleneck in training accurate prediction models. In addition, inconsistent terminology across commonly used datasets hinders effective model development and benchmarking. While recent advances have been driven largely by graph-based neural networks, the potential of sequence-based deep learning for p$K_a$ prediction remains underexplored. T5Chem, a sequence-based multitask chemical reaction model, offers an attractive way to cast molecular protonation/deprotonation as a language modeling task, and to couple microstate generation with subsequent p$K_a$ estimation. To pursue this direction, we introduce pKaCHU (p$K_a$ data that are Combined, Honed, and Updated), a curated dataset comprising 9,042 experimentally derived microscopic p$K_a$ entries with ionization-state annotations. We also present T5pKa, a text-based transformer model for small molecule p$K_a$ prediction built on T5Chem. T5pKa leverages multitask learning to enumerate microstates, enabling both protonation and deprotonation to be predicted by a single sequence-to-sequence model, and then predicts microscopic p$K_a$ values from the resulting microstate pairs using a separate regression model. Across benchmark datasets, T5pKa achieves performance comparable to established p$K_a$ prediction tools and published models, while offering the advantage of a unified multitasking framework for microstate enumeration and microscopic p$K_a$ prediction. 

# Installation

### Clone the repo
```bash
git clone https://github.com/charlotteinfante/t5pKa.git
cd t5pKa
```

### Create the environment 
```bash
conda create -n t5pKa python=3.9.10
conda activate t5pKa
pip install .
```

### Install Compatiable PyTorch
This repo requires Torch 1.7.1 (CUDA 11.0 build)
```bash
pip uninstall -y torch
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

# Run Predictions
Run single prediction:

**(A)** Predict protonated area of molecule

`$ python run_prediction.py --smiles "Prot:Brc1ccc(C2CN3C=CSC3=N2)cc1" --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv`

**(B)** Predict deprotonated area of molecule

`$ python run_prediction.py --smiles "Deprot:SMILES" --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv`

**(C)** Predict pKa of molecule

`$ python run_prediction.py --smiles "Brc1ccc(C2CN3C=CSC3=N2)cc1>>Brc1ccc(C2C[NH+]3C=CSC3=N2)cc1" --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv --scaler /path/to/scaler`

Run bulk prediction:

**(A)** Predict the ionization of more than one molecule

`$ python run_prediction.py --data_dir /path/to/data --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv`

**(B)** Predict the pKa of more than one molecule

`$ python run_prediction.py --data_dir /path/to/data --model_dir /path/to/model --prediction /path/to/prediction/prediction.csv --scaler /path/to/scaler`
