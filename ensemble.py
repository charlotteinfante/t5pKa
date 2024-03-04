'''
example for splitting the calculated data: python ensemble.py --raw_data /scratch/cii2002/t5chem_new/t5chem_prop/data/CHEMBL/SPLIT/basic/90_5_5/marvin/ 
--resulting_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/CHEMBL/SPLIT/ensemble/80_20/basic/

example for splitting the experimental data: python ensemble.py --raw_exp_data /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/8_1_1/basic/ 
--resulting_exp_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/ensemble/basic/

example for scaffold splitting data: python ensemble.py --all_data /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/Training/basic/ \
--resulting_scaffold_dir /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/scaffold/ensemble/basic/

example for getting the evalution of the ensemble model: python ensemble.py --prediction_dir /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/basic/ 
--prediction_file predictions_test.csv 
--prediction_file_save /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/basic/ensemble_prediction_internal_test.csv

'''
import os
import pandas as pd  
import numpy as np 
import rdkit 
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from data_utils import random_split
from scipy.stats import pearsonr, sem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        test_t = pd.read_csv(args.raw_exp_data+'test.target', names =['target'] )
        train = pd.concat([train_s, train_t], axis=1)
        val = pd.concat([val_s, val_t], axis=1)
        frames = [train, val]
        result = pd.concat(frames)
        ratio_train = len(train_t) / len(result)
        ratio_val = 1 - ratio_train
        for i in range(1,6):
            os.mkdir(str(args.resulting_exp_dir) + str(i))
            train, val, test = random_split(result, ratio_train, ratio_val, i)
            train['source'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/train.source', index = False, header=False)
            train['target'].to_csv(str(arg.sresulting_exp_dir)+str(i)+ '/train.target', index = False, header=False)
            test['source'].to_csv(str(arg.sresulting_exp_dir)+str(i)+ '/val.source', index = False, header=False)
            test['target'].to_csv(str(args.resulting_exp_dir)+str(i)+ '/val.target', index = False, header=False)

def scaffold_stratkfold_split(args):
    if args.all_data and args.resulting_scaffold_dir:
        os.makedirs(args.resulting_scaffold_dir)
        source = pd.read_csv(str(args.all_data)+'train.source', names = ['smiles'])
        target = pd.read_csv(str(args.all_data)+'train.target', names = ['targets'])
        data = pd.concat([source, target], axis=1)

        # get scaffold from each molecule 
        scaffold = [MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles) for smiles in data['smiles']]
        
        # get the folds based on scaffold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_indices = []
        val_indices = []
        for i, (train, val) in enumerate(cv.split(data['smiles'], scaffold)):
            train_indices.append(train)
            val_indices.append(val)
        
        for i in range(0,5):
            os.mkdir(str(args.resulting_scaffold_dir) + str(i+1))
            train_folds = data.iloc[train_indices[i]]
            val_folds = data.iloc[val_indices[i]]
            if i == 4:
                train_folds['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/train.source', index = False, header=False)
                train_folds['targets'].to_csv(str(args.resulting_scaffold_dir) +str(i+1)+'/train.target', index = False, header=False)
                val_folds['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/test.source', index = False, header=False)
                val_folds['targets'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/test.target', index = False, header=False)
            else:
                train_folds['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/train.source', index = False, header=False)
                train_folds['targets'].to_csv(str(args.resulting_scaffold_dir) +str(i+1)+'/train.target', index = False, header=False)
                val_folds['smiles'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/val.source', index = False, header=False)
                val_folds['targets'].to_csv(str(args.resulting_scaffold_dir)+str(i+1)+ '/val.target', index = False, header=False)

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

        print('RMSE:', mean_squared_error(test_1['target_0'], test_1['average'], squared=False))
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
