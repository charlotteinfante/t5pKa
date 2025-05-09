import argparse
from rdkit import Chem
import re
import numpy as np
import shap
import torch
from EFGs import mol2frag
from matplotlib import cm
from IPython.display import display, SVG, Image
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from model import T5ForProperty
from mol_tokenizers import SimpleTokenizer
#from t5chem import SimpleTokenizer, T5ForProperty
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
import matplotlib
from matplotlib.colors import Normalize
import os 
import joblib
import pdb


def add_args(parser):
    parser.add_argument(
        "--input_to_analyze",
        type=str,
        required=True,
        help="should be a rdkit canonical smiles string (seq2seq input) or similar to Brc1ccc(C2CN3C=CSC3=[NH+]2)cc1>>Brc1ccc(C2CN3C=CSC3=N2)cc1 in micropka input",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        required=True,
        help="path to the MinMaxScaler.gz file created during training /path/to/MinMaxScaler.gz",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="path to the model's directory",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="either regression or seq2seq",
    )

def analyze(args):
    def prettymol(smiles, shap_value,all_IDs, asMol=False, label=None, path='', imgsize=(300, 200), highlights=None):
        '''
        highlights is a dictionary, which may contain:
        highlightAtoms: list
        highlightBonds: list
        highlightAtomRadii: dict[int]=float, atom index (int), radius (float)
        highlightAtomColors: dict[int]=tuple, index (int), color (tuple, length=3)
        highlightBondColors: dict[int]=tuple,index (int), color (tuple, length=3)
        '''
        if asMol:
            mol = smiles.__copy__()
        else:
            mol = Chem.MolFromSmiles(smiles)
        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        if '.png' in path:
            drawer = rdMolDraw2D.MolDraw2DCairo(*imgsize)
        else:
            drawer = rdMolDraw2D.MolDraw2DSVG(*imgsize)
        opts = drawer.drawOptions()
    
        y = 0 
        # get the first idx of each functional group tuple
        first_atom_in_func_group = [t[0] for t in all_IDs]
        # link the first idx of each func group to each func group value and sort them from least to greatest
        paired_values = sorted(zip(shap_value, first_atom_in_func_group), key=lambda x: x[1])
        # get only the shap values from paired_values
        shap_value_organized = [t[0] for t in paired_values]
        # get the organized 0 to greatest first atom and make a list 
        first_atom_in_func_group = [t[1] for t in paired_values]
    
        if label == 'map':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(mol.GetAtomWithIdx(i).GetAtomMapNum())
        elif label == 'idx':
            for i in range(mol.GetNumAtoms()):
                opts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()+str(i)
        else:
            _ = [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
            for i in range(len(mol.GetAtoms())):
                atom = mol.GetAtomWithIdx(i)
                if y < len(first_atom_in_func_group) and i == first_atom_in_func_group[y]:
                    atom.SetProp('atomNote', str(round(shap_value_organized[y],2)))
                    print(round(shap_value_organized[y],2))
                    y += 1   
                else:
                    atom.SetProp('atomNote', '')
        if not highlights:
            drawer.DrawMolecule(mol)
        else:
            drawer.DrawMolecule(mol, **highlights)
        drawer.FinishDrawing()
        if '.png' in path:
            drawer.WriteDrawingText(path)
            display(Image(path))
        else:
            svg = drawer.GetDrawingText()
            display(SVG(svg.replace('svg:','')))
            if '.svg' in path:
                with open(path, 'w') as wf:
                    print(svg, file=wf)
        return drawer

    scaler = joblib.load(os.path.join(args.scaler))
    def classifier(x):
        x = [ i for i in x]
        #remove spaces made between each character 
        x = [s.replace(" " , "") for s in x]
        tv = tokenizer.batch_encode_plus(x, return_tensors='pt', truncation=True, padding="max_length", max_length=500)
        for k,v in tv.items():
            if isinstance(v, torch.Tensor):
                tv[k] = v.to(device)
        outputs = model(**tv).logits.detach().cpu().numpy()
        outputs_unscaled = scaler.inverse_transform(outputs)
        return outputs_unscaled

    def analysis(shap_value, molecule):
        '''prints out base value, sum of shap values, prediction from SHAP based on addition of base value and
        sum of shap values, and prediction from classifier function'''
        sum_array = shap_value.values.sum()
        sum_shap_base_value = sum_array + shap_values.base_values
        predictions = classifier([molecule])
        print("base value", shap_values.base_values)
        print("sum of SHAP",sum_array)
        print("SHAP pred", sum_shap_base_value)
        print("prediction from classifier", predictions)
        print('')

    # load the model and tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = SimpleTokenizer(vocab_file=args.model_dir + 'vocab.pt')
    if args.task_type == 'regression':
        model = T5ForProperty.from_pretrained(args.model_dir).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.config.task_specific_params = {}

    def map_atom2stridx(smiles):
        pattern = "[=#]?(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)[0-9]{0,2}"
        atom_finder = re.compile(pattern)
        ms = [x for x in atom_finder.finditer(smiles)]
        map_dict = {i:(x.start(),x.end()) for i,x in enumerate(ms)}
        smiles = smiles.replace('>>', '.') if '>>' in smiles else smiles.replace('>', '.')
        assert Chem.MolFromSmiles(smiles).GetNumAtoms() == len(map_dict)
        return map_dict
    def token2fg(fgids, map_dict):
        from itertools import chain
        fg_dict = {}
        for i,fg in enumerate(fgids):
            for a in fg:
                fg_dict[i] = list(chain(fg_dict.get(i,[]), range(map_dict[a][0],map_dict[a][1])))
        return fg_dict
    def smiles2fgid(smiles):
        '''
        Groups the SMILES string into functional groups.
        Input:
            SMILES string
        Returns:
            token2fg(fg_id+fgch_id, map_dict)
                Type: dictionary
                Key is within range of 0 to length of number of functional groups found.
                For each Key, is a list with the atom index of each atom in functional group 
            fg+chs
                Type: list 
                List of strings; a string representation of each functional group
            fg_id+fgch_id
                Type: list 
                List of tuples; containing each atom index in each functional group grouped by tuples
        example:
            fg_dict, all_fgs, all_ids = smiles2fgid('NC(Cc1cnc[nH]1)C(=O)O')
        '''
        map_dict = map_atom2stridx(smiles)
        mol = Chem.MolFromSmiles(smiles.replace('>>', '.') if '>>' in smiles else smiles.replace('>', '.'))
        fg, chs, fg_id, fgch_id = mol2frag(mol, returnidx=True, TreatHs='include')
        return token2fg(fg_id+fgch_id, map_dict), fg+chs, fg_id+fgch_id

    full_input = args.input_to_analyze.strip()
    data = [full_input]

    masker = shap.maskers.Text(tokenizer, mask_token="<pad>", collapse_mask_token=True)
    explainer = shap.Explainer(classifier, masker, output_names=['Property'])
    shap_values = explainer(data)

    # SHAP info
    sum_array = shap_values.values.sum()
    sum_shap_base_value = sum_array + shap_values.base_values
    predictions = classifier([full_input])
    print([
        'sum_array: ' + str(sum_array), 
        'sum_shap_base_value: ' + str(sum_shap_base_value.flatten().tolist()), 
        'predictions: ' + str(predictions)
    ])

    tokens = shap_values.data[0]  
    values = shap_values.values[0]  
    token_value_list = [(t, float(v)) for t, v in zip(tokens, values)]
    print(token_value_list)

    # split input if >> present to save 2 images
    if '>>' in args.input_to_analyze:
        delimiter_index = args.input_to_analyze.find('>>')
        smiles_list = args.input_to_analyze.split('>>')
    else:
        smiles_list = [args.input_to_analyze]

    for i, smiles in enumerate(smiles_list):
        print("=== Analyzing molecule " + str(i+1) + " ===")
        data = [smiles]

        # get functional groups from smiles
        fg_dict, all_fgs, all_ids = smiles2fgid(smiles)
        mol = Chem.MolFromSmiles(smiles)
        bwr = matplotlib.colormaps.get_cmap('bwr')

        # find the index of where ">>" is present
        if '>>' in args.input_to_analyze:
            if i == 0:
                shap_tokens = np.array([shap_values.values[0][x][0] for x in range(0,delimiter_index)])
            else:
                shap_tokens = shap_values.values[0][delimiter_index+2:, 0]
        else:
            shap_tokens = np.array([shap_values.values[0][i][0] for i in range(len(smiles))])
        
        for j, fg in enumerate(all_ids):
            if all_fgs[j] == 'O' and smiles[fg_dict[j][0] -1] == '(':
                fg_dict[j].append(fg_dict[j][0] + 1)
                fg_dict[j].append(fg_dict[j][0] - 1)
            if all_fgs[j] == '[NH3+]' and smiles[fg_dict[j][0] -1] == '(':
                first_value = fg_dict[j][0]
                last_value = fg_dict[j][-1]
                fg_dict[j].append(first_value - 1)
                fg_dict[j].append(last_value + 1)
            if all_fgs[j] == 'O=CO' and fg_dict[j][-1] + 1 < len(smiles) and smiles[fg_dict[j][-1] + 1] == ')':
                atom_of_interest = fg_dict[j][-1]
                fg_dict[j].append(atom_of_interest - 1)
                fg_dict[j].append(atom_of_interest + 1)
            if all_fgs[j] == 'NC=O' and smiles[fg_dict[j][-1]+1] == ')':
                oxygen = fg_dict[j][-1]
                d_bond = fg_dict[j][-1] - 1
                fg_dict[j].append(d_bond - 1)
                fg_dict[j].append(oxygen + 1)
            if all_fgs[j] == 'C=O' and smiles[fg_dict[j][-1]+1] == ')':
                oxygen = fg_dict[j][-1]
                d_bond = fg_dict[j][-1] - 1
                fg_dict[j].append(d_bond - 1)
                fg_dict[j].append(oxygen + 1)

        func_group_value = [np.sum(shap_tokens[fg_dict[j]]) for j in range(len(all_ids))]
        vmin = min(func_group_value)
        vmax = max(func_group_value)
        absmax = max(abs(vmin), abs(vmax))
        my_norm = Normalize(vmin=-absmax, vmax=absmax)

        color_dicts_a, color_dicts_b = {}, {}
        for j, fg in enumerate(all_ids):
            shap_val = shap_tokens[fg_dict[j]]
            print(all_fgs[j], shap_val, np.sum(shap_val))
            for a in fg:
                color_dicts_a[a] = bwr(my_norm(np.sum(shap_val)))

        for bond in mol.GetBonds():
            a = bond.GetBeginAtom().GetIdx()
            b = bond.GetEndAtom().GetIdx()
            if a in color_dicts_a and b in color_dicts_a and color_dicts_a[a] == color_dicts_a[b]:
                color_dicts_b[bond.GetIdx()] = color_dicts_a[a]

        prettymol(
            mol,
            func_group_value,
            all_ids,
            asMol=True,
            highlights={
                'highlightAtoms': color_dicts_a.keys(),
                'highlightAtomColors': color_dicts_a,
                'highlightBonds': color_dicts_b.keys(),
                'highlightBondColors': color_dicts_b
            },
            imgsize=(300, 300),
            path=f"shap_{i+1}_{smiles}.svg"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    analyze(args)