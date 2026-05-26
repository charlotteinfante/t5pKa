'''
example for splitting the calculated data: python ensemble.py --raw_data /scratch/cii2002/t5chem_new/t5chem_prop/data/CHEMBL/SPLIT/basic/90_5_5/marvin/ 
--resulting_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/CHEMBL/SPLIT/ensemble/80_20/basic/

example for splitting the experimental data: python ensemble.py --raw_exp_data /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/8_1_1/basic/ 
--resulting_exp_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/ensemble/basic/

example for scaffold splitting data: python ensemble.py --all_data /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/8_1_1/acidic_basic/ \
--resulting_scaffold_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/scaffold/ensemble/acidic_basic/

example for getting the evalution of the ensemble model: python ensemble.py --prediction_dir /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/basic/ 
--prediction_file predictions_test.csv 
--prediction_file_save /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/basic/ensemble_prediction_internal_test.csv

'''
import os
import re
import pdb
import pandas as pd  
import numpy as np 
import rdkit 
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils import random_split, make_canonical_smiles
from scipy.stats import pearsonr, sem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from t5pka.compat import rmse
except ImportError:
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


def add_args(parser):
    parser.add_argument(
        "--raw_data",
        type=str,
        required=False,
        help="should contain smiles and targets in same dataframe",)
    parser.add_argument(
        "--resulting_dir",
        type=str,
        required=False,
        help="path where output data should be"
    )
    parser.add_argument(
        "--raw_exp_data",
        type=str,
        required=False,
        help="the experimental data used to finetune the model"
    )
    parser.add_argument(
        "--resulting_exp_dir",
        type=str,
        required=False,
        help="the location of where you want the newly split experimental data to be"
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=False,
        help="the location of where ensemble models are"
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        required=False,
        help="name of the prediction file. example: predictions_test.csv"
    )
    parser.add_argument(
        "--prediction_file_save",
        type=str,
        required=False,
        help="path of where you want to save the prediction file with all the ensemble model predictions"
    )
    parser.add_argument(
        "--all_data",
        type=str,
        required=False,
        help="path of where data is (contains train.source and train.test only)"
    )
    parser.add_argument(
        "--resulting_scaffold_dir",
        type=str,
        required=False,
        help="path of where you want the newly scaffold splitted data in"
    )

def split_data_using_kfold(args):
     if args.resulting_dir and args.raw_data:
        os.makedirs(args.resulting_dir)
        train_s = pd.read_csv(str(args.raw_data)+'train.source', names = ['source'])
        val_s = pd.read_csv(args.raw_data+'val.source', names =['source'] )
        train_t = pd.read_csv(args.raw_data+'train.target', names = ['target'])
        val_t = pd.read_csv(args.raw_data+'val.target', names =['target'] )
        train = pd.concat([train_s, train_t], axis=1)
        val = pd.concat([val_s, val_t], axis=1)
        frames = [train, val]
        result = pd.concat(frames)

        #get folds using sklearn 
        kf = KFold(n_splits=5, shuffle = True,random_state= 42)
        train_indices = []
        val_indices = []
        for i,(train, val) in enumerate(kf.split(result)):
            train_indices.append(train)
            val_indices.append(val)
        for i in range(1,6):
            os.mkdir(str(args.resulting_dir) + str(i))
        for i in range(0,5):
            train_folds = result.iloc[train_indices[i]]
            val_folds = result.iloc[val_indices[i]]
            train_folds['source'].to_csv(str(args.resulting_dir)+str(i+1)+ '/train.source', index = False, header=False)
            train_folds['target'].to_csv(str(args.resulting_dir) +str(i+1)+'/train.target', index = False, header=False)
            val_folds['source'].to_csv(str(args.resulting_dir)+str(i+1)+ '/val.source', index = False, header=False)
            val_folds['target'].to_csv(str(args.resulting_dir)+str(i+1)+ '/val.target', index = False, header=False)
        else:
            print("")
            
def split_experimental_data(args):
    if args.resulting_exp_dir and args.raw_exp_data:
        os.makedirs(args.resulting_exp_dir)
        train_s = pd.read_csv(str(args.raw_exp_data)+'train.source', names = ['source'])
        val_s = pd.read_csv(args.raw_exp_data+'val.source', names =['source'] )
        train_t = pd.read_csv(args.raw_exp_data+'train.target', names = ['target'])
        val_t = pd.read_csv(args.raw_exp_data+'val.target', names =['target'] )
        train = pd.concat([train_s, train_t], axis=1)
        val = pd.concat([val_s, val_t], axis=1)
        frames = [train, val]
        result = pd.concat(frames)
        # get the same splitting ratio as baseline model 
        ratio_train = len(train_t) / len(result)
        ratio_val = 1 - ratio_train
        for i in range(1,6):
            os.mkdir(str(args.resulting_exp_dir) + str(i))
            train, val, test = random_split(result, ratio_train, ratio_val, i)
            train['source'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/train.source', index = False, header=False)
            train['target'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/train.target', index = False, header=False)
            test['source'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/val.source', index = False, header=False)
            test['target'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/val.target', index = False, header=False)

def scaffold_stratkfold_split(args):
    breakpoint()
    if args.all_data and args.resulting_scaffold_dir:
        os.makedirs(args.resulting_scaffold_dir)
        train_s = pd.read_csv(str(args.all_data)+'train.source', names = ['smiles'])
        val_s = pd.read_csv(args.all_data+'val.source', names =['smiles'] )
        train_t = pd.read_csv(args.all_data+'train.target', names = ['targets'])
        val_t = pd.read_csv(args.all_data+'val.target', names =['targets'] )
        #combine training and validation together 
        train = pd.concat([train_s, train_t], axis=1)
        val = pd.concat([val_s, val_t], axis=1)
        frames = [train, val]
        data = pd.concat(frames)
        data = data.reset_index()
        data = data.drop(['index'], axis=1)
        # get length of fold 
        smiles = data['smiles'].values
        length_data = len(smiles)
        fold_size = length_data // 5

        # condition for mixed dataset (has "acidic:" or "basic:" prefixes)
        if data['smiles'].str.contains('Pairs:|acidic:').any():
            data['prefix'] = [x.split(':')[0] for x in data['smiles']]
            data['placeholder'] = [x.split(':')[-1] for x in data['smiles']]
            data['first'] = [x.split('>>')[0] for x in data['placeholder']]
            data['second'] = [x.split('>>')[-1] for x in data['placeholder']]
            scaffold = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles) for smiles in data['first']]
        elif data['smiles'].str.contains('acidic:|basic:').any():
            data['just smiles'] = [x.split(':')[-1] for x in data['smiles']]
            # get scaffold from each molecule 
            scaffold = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles) for smiles in data['just smiles']]
        elif data['smiles'].str.contains('>>').any():
            data['first'] = [x.split('>>')[0] for x in data['smiles']]
            data['second'] = [x.split('>>')[-1] for x in data['smiles']]
            # get scaffold from each molecule 
            scaffold = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles) for smiles in data['first']]
        else:
            # get scaffold from each molecule 
            scaffold = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smile) for smile in smiles]

        # this chunk of code slightly modified from sklearn's straitedkfold function 
        unique_scaffold, unique_scaffold_idx, scaffolds = np.unique(scaffold,return_index=True, return_inverse=True)
        _, class_perm = np.unique(unique_scaffold_idx, return_inverse=True)
        scaffold_encoded = class_perm[scaffolds]
        num_of_unique_scaffolds = len(unique_scaffold_idx)

        # combine the encoded scaffold values with the actual data
        combined_array = np.column_stack((smiles, data['targets'].values, scaffold_encoded))
        # sort them based on the encoded values 
        sorted_array = combined_array[np.argsort(combined_array[:, 2])]
        # chunk up the data based on the similar encoded values 
        scaffold_values = sorted_array[:, 2]
        scaffold_change_indices = np.where(scaffold_values[:-1] != scaffold_values[1:])[0] + 1
        chunks = np.split(sorted_array, scaffold_change_indices)

        # do 5 kfold split based on chunk size 
        breakpoint()
        num_folds = 5
        fold_1, fold_2, fold_3, fold_4, fold_5 = [], [],[],[],[]
        for chunk in chunks:
            if len(fold_1) + len(chunk) <= fold_size:
                fold_1.extend(chunk)  
            elif len(fold_2) + len(chunk) <= fold_size:
                fold_2.extend(chunk)
            elif len(fold_3) + len(chunk) <= fold_size:
                fold_3.extend(chunk)
            elif len(fold_4) + len(chunk) <= fold_size:
                fold_4.extend(chunk)
            elif len(fold_5) + len(chunk) <= fold_size:
                fold_5.extend(chunk)
            
        fold_1 = np.array(fold_1)
        fold_2 = np.array(fold_2)
        fold_3 = np.array(fold_3)
        fold_4 = np.array(fold_4)
        fold_5 = np.array(fold_5)
        
        # code after this commented chunk does this in a loop 
            #train_1 = np.concatenate((fold_2, fold_3, fold_4, fold_5), axis=0)
            #val_1 = fold_1
            #train_2 = np.concatenate((fold_3, fold_4, fold_5, fold_1), axis=0)
            #val_2 = fold_2
            #train_3 = np.concatenate((fold_4, fold_5, fold_1, fold_2), axis=0)
            #val_3 = fold_3
            #train_4 = np.concatenate((fold_5, fold_1, fold_2, fold_3), axis=0)
            #val_4 = fold_4
        folds = [fold_1, fold_2, fold_3, fold_4, fold_5]
        num_folds = len(folds)
        train, val = [], []
        for i in range(num_folds):
            train.append(np.concatenate(folds[i+1:] + folds[:i], axis=0))
            val.append(folds[i])
        
        # extract needed info from lists and save them into files
        for i in range(0,5):
            os.makedirs(str(args.resulting_scaffold_dir) + str(i+1))
            val_list = val[i]
            train_list = train[i]
            column_labels = ['smiles', 'targets', 'unique_id']
            val_df = pd.DataFrame(val_list, columns=column_labels)
            train_df = pd.DataFrame(train_list, columns=column_labels)
            train_df['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/train.source', index = False, header=False)
            train_df['targets'].to_csv(str(args.resulting_scaffold_dir) +str(i+1)+'/train.target', index = False, header=False)
            val_df['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/val.source', index = False, header=False)
            val_df['targets'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/val.target', index = False, header=False)


def evaluate_ensemble(args):
    if args.prediction_dir and args.prediction_file and args.prediction_file_save:
        test_1 = pd.read_csv(args.prediction_dir + '1/chembl_datawarrior/' + args.prediction_file)
        for i in range(2,6):
            test = pd.read_csv(args.prediction_dir+str(i)+'/chembl_datawarrior/' + args.prediction_file)
            test_1['prediction_'+str(i-1)] = test['prediction_0']

        # calc the art. mean 
        avg = []
        for i in range(len(test_1)):
            avg.append(test_1[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].iloc[i].mean())
        # calc the standard deviation 
        std = []
        for i in range(len(test_1)):
            std.append(test_1[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].iloc[i].sem())
        test_1['average'] = avg 
        test_1['STDev'] = std 

        test_1.to_csv(args.prediction_file_save, index=False)
        r_value, prob = pearsonr(test_1['target_0'], test_1['average'])

        print('RMSE:', rmse(test_1['target_0'], test_1['average']))
        print('MAE:', mean_absolute_error(test_1['target_0'], test_1['average']))
        print('r2:', r2_score(test_1['target_0'], test_1['average']))
        print('r:', r_value)
    else:
        print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    split_data_using_kfold(args)
    split_experimental_data(args)
    scaffold_stratkfold_split(args)
    evaluate_ensemble(args)
