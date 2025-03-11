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
        "--scaler",
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
        "--num_beams",
        default=10,
        type=int,
        help="Number of beams for beam search.",
    )
    parser.add_argument(
        "--num_preds",
        default=5,
        type=int,
        help="The number of independently computed returned sequences for each element in the batch.",
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
    lg = rdkit.RDLogger.logger()  
    lg.setLevel(rdkit.RDLogger.CRITICAL) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(args.data_dir):
        args.data_dir, base = os.path.split(args.data_dir)
        base = base.split('.')[0]
    else:
        base = "test"

    # make sure the inputs are rdkit canonical SMILES 
    with open(args.data_dir +'/'+ base+ '.source', 'r') as smiles_file:
        smiles_list = [line.rstrip() for line in smiles_file]
    smiles_df = pd.DataFrame(smiles_list, columns=['source'])
    smiles_file.close()
    if smiles_df['source'].str.contains('acidic:|basic:').any():
        smiles_df[['prefix','just_smiles']] = smiles_df['source'].str.split(':',expand=True)
        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), canonical=True) for mol in smiles_df['just_smiles']]
        smiles_df['canonical'] = canonical
        smiles_df['combined'] = smiles_df[['prefix','canonical']].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
        smiles_df['combined'].to_csv(args.data_dir + 'test.source', index=False, header=False)   

    # option to chose the best checkpoint to run prediction (read in config)
    if args.best_cp == True:
        best_files = glob.glob(args.model_dir + '/best_*/')
        config = T5Config.from_pretrained(best_files[0])
    else:
        config = T5Config.from_pretrained(args.model_dir)

    # get task type information
    task = T5ChemTasks[config.task_type]

    # get the type of Tokenizer needed 
    tokenizer_type = getattr(config, "tokenizer")
    tokenizer_map = {"simple": SimpleTokenizer,"atom": AtomTokenizer, "selfies": SelfiesTokenizer }
    Tokenizer = tokenizer_map.get(tokenizer_type)
    tokenizer = Tokenizer(vocab_file=os.path.join(args.model_dir, 'vocab.pt'))

    #if os.path.isfile(args.data_dir):
    #    args.data_dir, base = os.path.split(args.data_dir)
    #    base = base.split('.')[0]
    #else:
    #    base = "test"

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

    if task.output_layer == 'seq2seq':
        targets = []
        task_specific_params = {
            "Reaction": {
            "early_stopping": True,
            "max_length": task.max_target_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_preds,
            "decoder_start_token_id": tokenizer.pad_token_id,
            }
        }
        # allow possibility to predict using best checkpoint
        if args.best_cp == True:
            best_files = glob.glob(args.model_dir +'/best_*/')
            model = T5ForConditionalGeneration.from_pretrained(best_files[0])
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        model.eval()
        model = model.to(device)

        with open(os.path.join(args.data_dir, base+".target")) as rf:
            for line in rf:
                targets.append(standize(line.strip()[:task.max_target_length]))
        test_df = pd.DataFrame(targets, columns=['target']) #added 
        predictions = [[] for i in range(args.num_preds)]
        for batch in tqdm(test_loader, desc="prediction"):
            for k, v in batch.items():
                batch[k] = v.to(device)
            del batch['labels']
            with torch.no_grad():
                outputs = model.generate(**batch, **task_specific_params['Reaction'])
            for i,pred in enumerate(outputs):
                prod = tokenizer.decode(pred, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                predictions[i % args.num_preds].append(prod)

    else:
        num_targets = testset[0]['decoder_input_ids'].shape[-1]
        predictions = np.zeros((len(testset), num_targets))
        targets = np.zeros_like(predictions)
        if args.best_cp == True:
            best_files = glob.glob(args.model_dir+'/best_*/')
            model = T5ForProperty.from_pretrained(best_files[0])
        else:
            model = T5ForProperty.from_pretrained(args.model_dir)
        model.eval()
        model = model.to(device)
        test_df = pd.DataFrame(targets, columns=['target_'+str(i) for i in range(num_targets)]) #added 

        for i, batch in enumerate(tqdm(test_loader, desc="prediction")):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**batch) 
                cur_start = i*args.batch_size 
                cur_end = cur_start + outputs.logits.shape[0]
                targets[cur_start : cur_end] = batch['labels'].detach().cpu().numpy()
                if task.output_layer == 'regression':
                    logits = outputs.logits 
                elif outputs.logits.shape[-1] == 2: # binary classification
                    logits = outputs.logits[:,-1:]
                    binary_classification = True
                else:
                    logits = torch.argmax(outputs.logits, axis=-1, keepdim=True)
                    binary_classification = False
                predictions[cur_start : cur_end] = logits.detach().cpu().numpy()

    if isinstance(predictions, list): 
        for i, preds in enumerate(predictions):
            test_df['prediction_{}'.format(i + 1)] = preds
            test_df['prediction_{}'.format(i + 1)] = \
                test_df['prediction_{}'.format(i + 1)].apply(standize)
        test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', args.num_preds), axis=1)

        correct = 0
        invalid_smiles = 0
        for i in range(1, args.num_preds+1):
            correct += (test_df['rank'] == i).sum()
            invalid_smiles += (test_df['prediction_{}'.format(i)] == '').sum()
            print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/len(test_df)*100, \
                invalid_smiles/len(test_df)/i*100))
    
    
    
    elif task.output_layer == 'regression':
        if args.scaler is not None:
            scaler = joblib.load(os.path.join(args.scaler,'MinMaxScaler.gz')) 
            predictions = scaler.inverse_transform(predictions)
        test_df[['prediction_'+str(i) for i in range(num_targets)]] = predictions # predictions_#
        MAE = mean_absolute_error(targets, predictions)      
        MSE = mean_squared_error(targets, predictions)
        if num_targets == 1: 
            LinResults = scipy.stats.linregress(predictions.reshape(-1), targets.reshape(-1))
            coef_determin = r2_score(targets.reshape(-1), predictions.reshape(-1))
            r_value, prob = pearsonr(targets.reshape(-1), predictions.reshape(-1))
            print("MAE: {}    RMSE: {}    r2: {}    r: {}".format(MAE, MSE**0.5, coef_determin, r_value))
        else:
            print("MAE: {}    RMSE: {}".format(MAE, MSE**0.5))
    
    
    
    else:
        if binary_classification:
            roc_auc = roc_auc_score(test_df['target_0'], predictions)
            test_df['prediction'] = predictions>0.5
            print('ROC-AUC: {:.3f}'.format(roc_auc), end='\t')
        else:
            test_df['prediction'] = predictions
            f1 = f1_score(test_df['target_0'], predictions, average='macro')
            print('F1 Score: {:.3f}'.format(f1), end='\t')
        test_df = test_df.astype(int)
        correct = sum(test_df['prediction'] == test_df['target_0'])
        print('Accuracy: {:.1f}%'.format(correct/len(test_df)*100))

    if not args.prediction:
        args.prediction = os.path.join(args.model_dir, 'predictions_'+base+'.csv')
    test_df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    predict(args)
