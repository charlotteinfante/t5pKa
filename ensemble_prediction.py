import argparse
import os
from functools import partial

import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
import scipy
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from data_utils import T5ChemTasks, TaskPrefixDataset, data_collator
from evaluation import get_rank, standize
from model import T5ForProperty
from mol_tokenizers import AtomTokenizer, SelfiesTokenizer, SimpleTokenizer
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import pearsonr
import glob
import pdb

def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )
    parser.add_argument(
        "--scaler_random",
        type=str,
        required=False,
        help="MinMaxScaler.gz file created during training. this argument takes in the path of where the training data is found",
    )
    parser.add_argument(
        "--scaler_scaffold",
        type=str,
        required=False,
        help="MinMaxScaler.gz file created during training. this argument takes in the path of where the training data is found",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="The model path to be loaded.",
    )
    parser.add_argument(
        "--prediction",
        default='',
        type=str,
        help="The file name for prediction.",
    )
    parser.add_argument(
        "--prefix",
        default='',
        type=str,
        help="When provided, use it instead of read from trained model. (Especially useful when trained on a mixed\
            dataset, but want to test on seperate tasks)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--best_cp",
        default=False,
        type=bool,
        help="predict the best step based on lowest loss",
    )

def predict(args):
    '''
    Evaluates and gives the prediction for the regression ensemble model.

    example input: python ensemble_prediction.py --prediction /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/acidic/predictions_novartis.csv \
    --data_dir /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/Novartis/acidic/ \
    --model_dir /scratch/cii2002/t5chem_new/t5chem_prop/model_ENSEMBLE_kfold/acidic/ \
    --scaler /scratch/cii2002/t5chem_new/t5chem_prop/data/TRAINING/SPLIT/ensemble/acidic/
    '''
    lg = rdkit.RDLogger.logger()  
    lg.setLevel(rdkit.RDLogger.CRITICAL) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make sure the inputs are rdkit canonical SMILES 
    with open(args.data_dir + 'test.source', 'r') as smiles_file:
        smiles_list = [line.rstrip() for line in smiles_file]
    smiles_df = pd.DataFrame(smiles_list, columns=['source'])
    smiles_file.close()
    if smiles_df['source'].str.contains('acidic:|basic:').any():
        smiles_df[['prefix','just_smiles']] = smiles_df['source'].str.split(':',expand=True)
        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), canonical=True) for mol in smiles_df['just_smiles']]
        smiles_df['canonical'] = canonical
        smiles_df['combined'] = smiles_df[['prefix','canonical']].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
        smiles_df['combined'].to_csv(args.data_dir + 'test.source', index=False, header=False)   


    all_predictions, all_targets = [], []
    #breakpoint()
    for i in range(1,11):
        # option to chose the best checkpoint to run prediction (read in config)
        if args.best_cp == True:
            best_files = glob.glob(args.model_dir + str(i) +'/best_*/')
            config = T5Config.from_pretrained(best_files[0])
        else:
            config = T5Config.from_pretrained(args.model_dir + str(i) )
        
        # get task type information
        task = T5ChemTasks[config.task_type]

        # get the type of Tokenizer needed 
        tokenizer_type = getattr(config, "tokenizer")
        tokenizer_map = {"simple": SimpleTokenizer,"atom": AtomTokenizer, "selfies": SelfiesTokenizer }
        Tokenizer = tokenizer_map.get(tokenizer_type)
        tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, str(i)+'/vocab.pt'))

        if os.path.isfile(args.data_dir):
            args.data_dir, base = os.path.split(args.data_dir)
            base = base.split('.')[0]
        else:
            base = "test"
        # get scaler 
        if 1 <= i <= 5:
            path = args.scaler_random,str(i)
            scaler = joblib.load(os.path.join(args.scaler_random,str(i)+'/MinMaxScaler.gz')) 
        else:
            scaler = joblib.load(os.path.join(args.scaler_scaffold,str(i - 5)+'/MinMaxScaler.gz')) 

        testset = TaskPrefixDataset(tokenizer, data_dir=args.data_dir,
                                    prefix=args.prefix or task.prefix,
                                    max_source_length=task.max_source_length,
                                    max_target_length=task.max_target_length,
                                    separate_vocab=(task.output_layer != 'seq2seq'),
                                    type_path=base)
        data_collator_padded = partial(data_collator, pad_token_id=tokenizer.pad_token_id)
        test_loader = DataLoader(
                testset, 
                batch_size=args.batch_size,
                collate_fn=data_collator_padded
        )

        num_targets = testset[0]['decoder_input_ids'].shape[-1]
        predictions = np.zeros((len(testset), num_targets))
        targets = np.zeros_like(predictions)
        if args.best_cp == True:
            best_files = glob.glob(args.model_dir + str(i) +'/best_*/')
            model = T5ForProperty.from_pretrained(best_files[0])
        else:
            model = T5ForProperty.from_pretrained(args.model_dir + str(i))
        model.eval()
        model = model.to(device)

        for i, batch in enumerate(tqdm(test_loader, desc="prediction")):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            with torch.no_grad():
                outputs = model(**batch) # Pull out a single example. Check if that works. 
                cur_start = i*args.batch_size #What does this do?
                cur_end = cur_start + outputs.logits.shape[0]
                targets[cur_start : cur_end] = batch['labels'].detach().cpu().numpy()
                logits = outputs.logits
                predictions[cur_start : cur_end] = logits.detach().cpu().numpy()
        all_targets.append(targets)
        predictions = scaler.inverse_transform(predictions)
        all_predictions.append(predictions)

    d = {'targets': all_targets[0].flatten()}
    df = pd.DataFrame(data=d)
    for i in range(len(all_predictions)):
        df['prediction_'+str(i)] = all_predictions[i]
    
    avg, std, scaffold_avg, scaffold_std, random_avg, random_std = [], [], [], [], [], []
    for i in range(len(df)):
        # calc the art. mean 
        random_avg.append(df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].iloc[i].mean())
        scaffold_avg.append(df[['prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9']].iloc[i].mean())
        avg.append(df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4',
        'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9']].iloc[i].mean())
        # calc the standard deviation 
        random_std.append(df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4']].iloc[i].sem())
        scaffold_std.append(df[['prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9']].iloc[i].sem())
        std.append(df[['prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4',
        'prediction_5', 'prediction_6', 'prediction_7', 'prediction_8', 'prediction_9']].iloc[i].sem())

    # add mean stan. dev to dataframe 
    df['average'] = avg 
    df['STDev'] = std 
    df['random split average'] = random_avg
    df['random split STDev'] = random_std   
    df['scaffold split average'] = scaffold_avg
    df['scaffold split STDev'] = scaffold_std

    r_value, prob = pearsonr(df['targets'], df['average'])

    print('RMSE:', mean_squared_error(df['targets'], df['average'], squared=False))
    print('MAE:', mean_absolute_error(df['targets'], df['average']))
    print('r2:', r2_score(df['targets'], df['average']))
    print('r:', r_value)
    print('RMSE random split:', mean_squared_error(df['targets'], df['random split average'], squared=False))
    print('RMSE scaffold split:', mean_squared_error(df['targets'], df['scaffold split average'], squared=False))

    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions_'+base+'.csv')
    else: 
        df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    predict(args)
