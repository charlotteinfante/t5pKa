import argparse
import os
import pickle
from rdkit import Chem
import pdb

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
    first = []
    second = []
    for x, y in data.items():
        for z, lt in y.items():
            if 'smiles_list' in z:
                first_ = map(lambda x: x[0],lt)
                second_ = map(lambda x: x[1], lt)
                for mol_first in first_:
                    first.append(mol_first)
                for mol_second in second_:
                    second.append(mol_second)

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

    # get formal charge of molecules to find net neutral molecules with atomic charges 
    charged_atoms_count = []
    charged_atoms_count_second = []
    for index, i in enumerate(first):
        mol_id = Chem.MolFromSmiles(i)
        formal_charge = Chem.GetFormalCharge(mol_id)
        charged_count = sum(1 for atom in mol_id.GetAtoms() if atom.GetFormalCharge() != 0)
        charged_atoms_count.append(charged_count)
    for index, i in enumerate(second):
        mol_id = Chem.MolFromSmiles(i)
        formal_charge = Chem.GetFormalCharge(mol_id)
        charged_count = sum(1 for atom in mol_id.GetAtoms() if atom.GetFormalCharge() != 0)
        charged_atoms_count_second.append(charged_count)

    # find the true neutrals that are the first molecules 
    check_mol = []
    for index, (first_molecule, second_molecule) in enumerate(zip(charged_atoms_count, charged_atoms_count_second)):   
        if first_molecule > second_molecule:
            check_mol.append(index)
    correct_first = [second[i] for i in check_mol]
    correct_second = [first[i] for i in check_mol]
    correct_pka = [pka_list[i] for i in check_mol]
    check_mol.sort(reverse=True)
    for i in check_mol:
        first.pop(i)
        second.pop(i)
        pka_list.pop(i)
    merge_first = first + correct_first
    merge_second = second + correct_second
    merge_pka = pka_list + correct_pka
    #make source and target files for deprotonated and protonated molecules, respectfully
    for mol_prot in merge_first:
        print(mol_prot, file=protonated_file)
    protonated_file.close()
    for mol_deprot in merge_second:
        print(mol_deprot, file=deprotonated_file)
    deprotonated_file.close()
    pka_saved_file = open(pka_file, 'w')
    for pka_value in merge_pka:
        print(pka_value, file=pka_saved_file)
    pka_saved_file.close()

    #write deprotonated and protonated molecules into one file as pairs
    pairs_saved_file = open(pairs_file, 'w')
    with open(prot_file, 'r') as f1, open(deprot_file, 'r') as f2:
        for prot_mol, deprot_mol in zip(f1, f2):
            prot_mol = prot_mol.strip()
            deprot_mol = deprot_mol.strip()
            print(prot_mol+'>>'+deprot_mol, file=pairs_saved_file)
    pairs_saved_file.close()

if __name__ == "__main__":
    main()    

