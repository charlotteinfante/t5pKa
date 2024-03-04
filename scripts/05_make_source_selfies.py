import argparse
import os
import pickle
import selfies as sf

'''
open pickle file to extract pkas and smiles molecule, while converting smiles 
molecules into selfies
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input filename, type: .pkl")
    parser.add_argument("--output_prot", help="output filename for protonated selfies molecues, type: .target")
    parser.add_argument("--output_deprot", help="output filename for deprotonated selfies molecules, type: .source")
    parser.add_argument("--output_pairs", help="output filename for pairs of selfies, type: .source")
    parser.add_argument("--output_pka", help="output filename for pka, type: .target")
    args = parser.parse_args()

    print("inputfile:", args.input)
    print("outputfile for protonated selfies:", args.output_prot)
    print("outputfile for deprotonated selfies:", args.output_deprot)
    print("outputfile for pairs of selfies:", args.output_pairs)
    print("outputfile for pka:", args.output_pka)

    pickle_file = args.input
    prot_file = args.output_prot
    deprot_file = args.output_deprot
    pairs_file = args.output_pairs
    pka_file = args.output_pka


    #open and load input pickle file
    with open(pickle_file, 'rb') as f:
        data=pickle.load(f)

    file_1 = open(prot_file, 'w')
    file_2 = open(deprot_file, 'w')
    selfies_prot = []
    selfies_deprot = []

    #sort through dict in pickle file and extract smiles and convert smiles to selfies 
    for x, y in data.items():
        for z, lt in y.items():
            if 'smiles_list' in z:
                prot = map(lambda x: x[0],lt)
                deprot = map(lambda x: x[1], lt)
                for mol_prot in prot:
                    selfies_prot.append(sf.encoder(mol_prot))
                for mol_deprot in deprot:
                    selfies_deprot.append(sf.encoder(mol_deprot))

    #write deprotonated and protonated selfies into two separate files 
    for mol_prot in selfies_prot:
        print(mol_prot, file=file_1)
    for mol_deprot in selfies_deprot:
        print(mol_deprot, file=file_2)
    file_1.close()
    file_2.close()
    
    #write deprotonated and protonated selfies into one file as pairs 
    file_3 = open(pairs_file, 'w')
    with open(prot_file, 'r') as f1, open(deprot_file, 'r') as f2:
        for prot_mol, deprot_mol in zip(f1, f2):
            prot_mol = prot_mol.strip()
            deprot_mol = deprot_mol.strip()
            print(deprot_mol+'>>'+prot_mol, file=file_3)
    file_3.close()

    #extract pKa values from pickle file and write it into a target file
    file_4 = open(pka_file, 'w')
    for x, y in data.items():
        for z, lt in y.items():
            if 'pKa_list' in z:
                print(*lt, sep='\n', file=file_4)
    file_4.close()

if __name__ == "__main__":
    main()    

