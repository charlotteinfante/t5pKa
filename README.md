# T5Chem_pKa
A Unified Deep Learning Method for p<i>K</i><sub><i>a</i></sub> and Protonation prediction
<img width="736" alt="Screenshot 2024-03-14 at 3 24 27 PM" src="https://github.com/charlotteinfante/t5chem_pKa/assets/96793416/361bbd06-4c5e-4cda-b709-3f14ebc16a0c">


Inspired by the work of Jieyu Lu et al. (2022) {https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01467}, we use T5Chem--a T5 model built on HuggingFace Transformers--to predict macroscopic p<i>K</i><sub><i>a</i></sub>, mircoscopic p<i>K</i><sub><i>a</i></sub>, and protonation sites of small molecules. 

## Models
We leverage the multitasking ability of T5Chem, and we added the prefixes "acidic" and "basic" to train __one__ macroscopic p<i>K</i><sub><i>a</i></sub> model. Depending on the type of p<i>K</i><sub><i>a</i></sub> the user would like to predict, they can attach the prefix to the SMILES of the molecule of interest to predict either its acidic or basic macroscopic p<i>K</i><sub><i>a</i></sub>. 

Similarly, we reused the prefixes "Product" and "Reactants" from the original T5Chem to train our seq2seq model. Here, the "Product" prefix works for wanting to predict the area of deprotonation of the molecule, and the "Reactants" prefix works for predicting the protonation of the molecule. 

Lastly, our microscopic p<i>K</i><sub><i>a</i></sub> model uses an input of a molecule and its deprotonated or protonated joined together by the delimiter ">>". This model also is unique in a case where the user does not know the area of ionization, then they can use the seq2seq model output and join it with their seq2seq model input and predict the microscopic p<i>K</i><sub><i>a</i></sub>. The image below depicts the different prefixes and an example of inputs depending on the model: (de)protonation prediction (green), mircoscopic p<i>K</i><sub><i>a</i></sub> prediction (pink), and macrosopic p<i>K</i><sub><i>a</i></sub> prediction (yellow and blue). 

<img width="552" alt="Screenshot 2024-03-13 at 5 33 37 PM" src="https://github.com/charlotteinfante/t5chem_pKa/assets/96793416/3877e501-be9b-497d-9225-f632cd2f74c3">

## Enviroment / Set up

## Running a prediction
### To be edited still (usable only for Song and others with access to my folders)
To run a (de)protonation prediction using the seq2seq model
```bash
python __main__.py predict --data_dir /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/SAMPL6/seq2seq/mixed --model_dir /scratch/cii2002/t5chem_new/t5chem_pKa/models/seq2seq --prediction models/seq2seq/sampl6.csv
```
To run a microscopic p<i>K</i><sub><i>a</i></sub> prediction using the ensemble model
```bash
python ensemble_prediction.py --scaler_random /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/ensemble/pairs/mix/no_labels --scaler_scaffold /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/scaffold/ensemble/pairs/mix/no_labels/ --data_dir /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/SAMPL6/pairs/mix/no_labels/ --model_dir /scratch/cii2002/t5chem_new/t5chem_pKa/models/microscopic/
--prediction models/microscopic/sampl6.csv
```
To run a macroscopic p<i>K</i><sub><i>a</i></sub> prediction using the ensemble model
```bash
python ensemble_prediction.py --scaler_random /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/ensemble/acidic_basic/ --scaler_scaffold /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/scaffold/ensemble/acidic_basic/ --data_dir /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/SAMPL6/macropka/regression_monoprotic/acidic_basic/ --model_dir /scratch/cii2002/t5chem_new/t5chem_pKa/models/macroscopic/ --prediction models/macroscopic/sampl6.csv

```

## Training the models
Calculated p<i>K</i><sub><i>a</i></sub> datasets using Epik and pKasolver scripts (used for the seq2seq and micropscopic models)
