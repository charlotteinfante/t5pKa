# Installation

$ pip install t5chem
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
