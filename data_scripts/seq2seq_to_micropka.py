import pandas as pd
import argparse
from rdkit import Chem
import pdb

'''
example input: python seq2seq_to_micropka.py --seq2seq_input ../../pka/data/SAMPL8/unified/seq2seq/test.source --seq2seq_output ../../model_DEPROT/mixed/finetune/sampl8.csv  \
--regression_targets ../../pka/data/SAMPL8/pairs/test.target --save ../../pka/data/SAMPL8/unified/micropka_regression/
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq2seq_input",
        type=str,
        required=True,
        help="path to directory that contains inputs of seq2seq model (.source file)",
    )
    parser.add_argument(
        "--seq2seq_output",
        type=str,
        required=True,
        help="path to directory that contains results from seq2seq model .csv file",
    )
    parser.add_argument(
        "--regression_targets",
        type=str,
        required=True,
        help="path to directory that contains targets needed in regression task .csv file",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="path to directory to save .source and .target files for input into regression model",
    )

    args = parser.parse_args()

    result = pd.read_csv(args.seq2seq_output)
    inputs = pd.read_csv(args.seq2seq_input, names=['source'])
    targets = pd.read_csv(args.regression_targets, names=['target'])
    inputs[['prefix','smiles']] = inputs['source'].str.split(':',expand=True)

    pairs, skip_indices = [], []
    for i, (x, y, z) in enumerate(zip(inputs['smiles'], result['prediction_1'], result['prediction_2'])):
        # choose prediction: prefer y if valid, otherwise z
        if pd.notna(y):
            pred = y
        elif pd.notna(z):
            pred = z
        else:
            skip_indices.append(i)
            continue  # skip if both are NaN

        mol1 = Chem.MolFromSmiles(str(x))
        mol2 = Chem.MolFromSmiles(str(pred))

        # skip invalid SMILES
        if mol1 is None or mol2 is None:
            skip_indices.append(i)
            continue

        # compute formal charges
        charge1 = sum(atom.GetFormalCharge() for atom in mol1.GetAtoms())
        charge2 = sum(atom.GetFormalCharge() for atom in mol2.GetAtoms())

        # ensure correct direction
        if (charge1 == 1 and charge2 == 0) or (charge1 == 0 and charge2 == -1):
            pair = f"{Chem.MolToSmiles(mol1)}>>{Chem.MolToSmiles(mol2)}"
        elif (charge2 == 1 and charge1 == 0) or (charge2 == 0 and charge1 == -1):
            pair = f"{Chem.MolToSmiles(mol2)}>>{Chem.MolToSmiles(mol1)}"
        elif charge1 > charge2:
            pair = f"{Chem.MolToSmiles(mol1)}>>{Chem.MolToSmiles(mol2)}"
        elif charge1 < charge2:
            pair = f"{Chem.MolToSmiles(mol2)}>>{Chem.MolToSmiles(mol1)}"
        else:
            # not a +1↔0 or 0↔−1 transition
            skip_indices.append(i)
            continue

        pairs.append(pair)

    # drop skipped rows from targets
    targets_clean = targets.drop(skip_indices).reset_index(drop=True)

    # make sure pairs and targets are aligned
    df = pd.DataFrame({'source': pairs, 'target': targets_clean['target']})

    print(f"Skipped {len(skip_indices)} rows due to missing predictions or invalid charge transitions.")

    df['source'].to_csv(str(args.save)+"test.source", index=False, header=False)
    df['target'].to_csv(str(args.save)+"test.target", index=False, header=False)
    
if __name__ == "__main__":
    main() 