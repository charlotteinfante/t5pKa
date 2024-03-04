import rdkit
from rdkit import Chem 
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd 
import argparse
import pdb
'''
example input: python acidic_basic_assignment.py --data /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/SAMPL6/macropka/regression_monoprotic/test.source \
--targets True --data_targets /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/SAMPL6/macropka/regression_monoprotic/test.target
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to directory that contains smiles file",
    )
    parser.add_argument(
        "--data_targets",
        type=str,
        required=False,
        help="path to directory that contains targets file",
    )
    parser.add_argument(
        "--targets",
        default=False,
        type=bool,
        help="set to True if your directory that has file with SMILES also has a separate file with targets aka \
        a test.source and test.target",
    )
    parser.add_argument(
        "--using_seq2seq_result",
        default=False,
        type=bool,
        help="set to True if using results from seq2seq model",
    )
    args = parser.parse_args()
    
    # read in smarts list by MolGpka
    #smarts = pd.read_table('smarts_pattern.tsv')
    #acid = smarts[smarts['Acid_or_base'] == 'A']
    #base = smarts[smarts['Acid_or_base'] == 'B']
    #acids = acid['    SMARTS'].values
    #bases = base['    SMARTS'].values

    smarts = pd.read_csv('acidic_basic_assignment_list.csv')
    acid = smarts[smarts['acid_or_base'] == 'A ']
    base = smarts[smarts['acid_or_base'] == 'B']
    acids = acid['smiles'].values
    bases = base['smiles'].values

    # acidic groups
    carboxylic_acid = 'C(=O)O'  # smarts rep: '[#6](=[#8])-[#8]'            
    phenol = 'c1ccc(O)cc1'      # smarts rep: '[#6]1:[#6]:[#6]:[#6](-[#8]):[#6]:[#6]:1'            
    sulfonic_acid = 'OS(=O)(=O)O' # smarts rep: '[#8]-[#16](=[#8])(=[#8])-[#8]'
    sulfonamide = 'NS(=O)(=O)N'  # smarts rep: '[#7]-[#16](=[#8])(=[#8])-[#7]'
    imide = 'C(=O)NC(=O)' #'*1C(=O)NC(=O)*1'  # smarts rep: '[#6](=[#8])-[#7]-[#6]=[#8]'
    phosphoric_acid = 'OP(=O)(O)O' # smarts rep: '[#8]-[#15](=[#8])(-[#8])-[#8]'
    #acids = [carboxylic_acid, phenol, sulfonic_acid, sulfonamide, imide, phosphoric_acid]

    #basic groups
    amino = 'C[N]' #'*[N]' #'[A]-[#7]' #'[NH2]'  # ***** specify this better ????? ******
    sec_amine = '[H]N([C])[C]' #'[#7H](-[A])-[A]' #'CNC' #'[H]N([*])[*]'
    tert_amine = '[C]N([C])[C]' #'[*]N([*])[*]'
    quart_amine = 'C[N](C)C' #'*[N](*)*'
    guanidine = 'NC(=N)N'
    imidazole = 'n1c[nH]cc1'
    pyridine = 'n1ccccc1'
    #bases = [ amino, sec_amine, tert_amine, quart_amine, guanidine, imidazole, pyridine]

    groups = {'acidic':acids, 'basic':bases}

    # empty list to capture group 
    acidic_mols, basic_mols, no_match = [], [], []

    # read in file with smiles molecules 
    data = pd.read_csv(str(args.data), names=['smiles'])
    if args.using_seq2seq_result == True:
        data = pd.read_csv(str(args.data))
        data['smiles'] = data['prediction_1']

    if args.targets == True:
        target = pd.read_csv(str(args.data_targets), names=['target'])
        data = pd.concat([data,target], axis=1)
    else:
        target = [0 for i in range(len(data))]
        data['target'] = target
    print(len((data)))

# in loop using a specific molecule from SAMPL6 ; later loop in each molecule from a file 
    for molecule_, target in zip(data['smiles'].values, data['target'].values):
        molecule = Chem.MolFromSmiles(molecule_)
        acidic_match = False
        basic_match = False 
        print(molecule_)
        breakpoint()
        for key, value in groups.items():
            for i in value:

                #mol_group = Chem.MolFromSmarts(i)
                #smarts = Chem.MolToSmarts(mol_group)
                #matches = molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts))
                mol_group = Chem.MolFromSmiles(i)
                #smarts = Chem.MolToSmarts(mol_group)
                #matches = molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts))
                matches = molecule.GetSubstructMatches(Chem.MolFromSmiles(i))
                if key == 'acidic' and len(matches) > 0:
                    acidic_match = True 
                    acidic_mols.append(('acidic:'+str(molecule_), target))
                    print(key, i, molecule_)
                if key == 'basic' and len(matches) > 0:
                    basic_match = True 
                    basic_mols.append(('basic:'+str(molecule_), target))
                    print(key, i, molecule_)
        if not acidic_match and not basic_match:
            no_match.append((molecule_, target))

    # in case there are duplicates, just drop it from lists 
    acidic_mols = list(set(acidic_mols))
    basic_mols = list(set(basic_mols))
    no_match = list(set(no_match))

    print(len(no_match))
    print('no match:',no_match)

    # separate tuples 
    acidic_smiles = [i[0] for i in acidic_mols]
    acidic_targets = [i[1] for i in acidic_mols]
    basic_smiles = [i[0] for i in basic_mols]
    basic_targets = [i[1] for i in basic_mols]

    print(acidic_smiles, acidic_targets)
    print(basic_smiles, basic_targets)

if __name__ == "__main__":
    main() 