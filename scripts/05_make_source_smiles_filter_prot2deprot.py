import argparse
import os
import pickle
from rdkit import Chem

'''
open pickle file to extract pkas and smiles molecule
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .pkl")
    parser.add_argument("--output_prot", help="output filename for protonated smiles molecues, type: .source")
    parser.add_argument("--output_deprot", help="output filename for deprotonated smiles molecules, type: .target")
    parser.add_argument("--output_pairs", help="output filename for pairs of smiles, type: .source")
    parser.add_argument("--output_pka", help="output filename for pka, type: .target")
    args = parser.parse_args()

    print("inputfile:", args.input)
    print("outputfile for protonated smiles:", args.output_prot)
    print("outputfile for deprotonated smiles:", args.output_deprot)
    print("outputfile for pairs of smiles:", args.output_pairs)
    print("outputfile for pka:", args.output_pka)

    pickle_file = args.input
    prot_file = args.output_prot
    deprot_file = args.output_deprot
    pairs_file = args.output_pairs
    pka_file = args.output_pka


    #open and load input pickle file
    with open(pickle_file, 'rb') as f:
        data=pickle.load(f)
    protonated_file = open(prot_file, 'w')
    deprotonated_file = open(deprot_file, 'w')

    #sort through dict in pickle file and extract smiles  
    prot = []
    deprot = []
    for x, y in data.items():
        for z, lt in y.items():
            if 'smiles_list' in z:
                prot_ = map(lambda x: x[0],lt)
                deprot_ = map(lambda x: x[1], lt)
                for mol_prot in prot_:
                    prot.append(mol_prot)
                for mol_deprot in deprot_:
                    deprot.append(mol_deprot)

    #extract pKa values from pickle file and write it into a target file
    pka_saved_file = open(pka_file, 'w')
    for x, y in data.items():
        for z, lt in y.items():
            if 'pKa_list' in z:
                print(*lt, sep='\n', file=pka_saved_file)
    pka_saved_file.close()

    pka_saved_file = open(pka_file, 'r')
    pka_list = []
    for line in pka_saved_file:
            pka_list.append(line.strip())
    pka_saved_file.close()

    #check if protonated and deprotonated molecules are ordered correctly
    prot_charge = []
    deprot_charge = []
        #get formal charges from protonated and deprotonated molecules
    for i in prot:
        mol_id = Chem.MolFromSmiles(i)
        prot_charge.append(Chem.GetFormalCharge(mol_id))
    for i in deprot:
        mol_id = Chem.MolFromSmiles(i)
        deprot_charge.append(Chem.GetFormalCharge(mol_id))

    check_mols = []
    for index, (deprotonated, protonated) in enumerate(zip(deprot_charge, prot_charge)):
        if deprotonated > protonated:
            check_mols.append(index)
    print('# of mols dropped', len(check_mols))

        #drop molecules that are not ordered correctly 
    check_mols.sort(reverse=True)
    for index in check_mols:
        prot.pop(index)
        deprot.pop(index)
        pka_list.pop(index)

    #make source and target files for deprotonated and protonated molecules, respectfully
    for mol_prot in prot:
        print(mol_prot, file=protonated_file)
    protonated_file.close()
    for mol_deprot in deprot:
        print(mol_deprot, file=deprotonated_file)
    deprotonated_file.close()
    pka_saved_file = open(pka_file, 'w')
    for pka_value in pka_list:
        print(pka_value, file=pka_saved_file)
    pka_saved_file.close()

    #write deprotonated and protonated molecules  into one file as pairs
    pairs_saved_file = open(pairs_file, 'w')
    with open(prot_file, 'r') as f1, open(deprot_file, 'r') as f2:
        for prot_mol, deprot_mol in zip(f1, f2):
            prot_mol = prot_mol.strip()
            deprot_mol = deprot_mol.strip()
            print(prot_mol+'>>'+deprot_mol, file=pairs_saved_file)
    pairs_saved_file.close()

if __name__ == "__main__":
    main()    

