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
        "--best_cp",
        default=False,
        type=bool,
        help="predict the best step based on lowest loss",
    )
    parser.add_argument(
        "--smiles",
        default='',
        type=str,
        help="single SMILES string input for prediction instead of file",
    )
    parser.add_argument(
        "--unify",
        default=False,
        type=bool,
        help="use seq2seq prediction results in regression model"
    )
    parser.add_argument(
        "--regression_model_dir",
        default=None,
        type=str,
        help="read in regression model when --unify True"
    )
    parser.add_argument(
        "--regression_targets",
        default='',
        type=str,
        help="if regression file has targets and you want to compute metrics, then add path to this target file"
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
    if smiles_df['source'].str.contains('Prot:|Deprot:').any():
        smiles_df[['prefix','just_smiles']] = smiles_df['source'].str.split(':',expand=True)
        canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), canonical=True) for mol in smiles_df['just_smiles']]
        smiles_df['canonical'] = canonical
        smiles_df['combined'] = smiles_df[['prefix','canonical']].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
        smiles_df['combined'].to_csv(args.data_dir + 'test.source', index=False, header=False)   

    if args.unify == True:
        # read in seq2seq task [one model only, so no call in for ensemble]
        if args.best_cp == True:
            best_files = glob.glob(args.model_dir + str(i) +'/best_*/')
            config = T5Config.from_pretrained(best_files[0])
        else:
            config = T5Config.from_pretrained(args.model_dir)
        # get task information [should be seq2seq]
        task = T5ChemTasks[config.task_type]

        # get the type of Tokenizer needed 
        tokenizer_type = getattr(config, "tokenizer")
        tokenizer_map = {"simple": SimpleTokenizer, "atom": AtomTokenizer, "selfies": SelfiesTokenizer}
        Tokenizer = tokenizer_map[tokenizer_type]
        tokenizer = Tokenizer(vocab_file = os.path.join(args.model_dir, 'vocab.pt'))
        # Handle input: single SMILES or file-based
        if args.smiles:
            base = "single"
            smiles_list = [args.smiles]
            targets_raw = None
        # read in test.source file 
        else:
            if os.path.isfile(args.data_dir):
                args.data_dir, base = os.path.split(args.data_dir)
                base = base.split('.')[0]
            else:
                base = "test"
            with open(os.path.join(args.data_dir, f"{base}.source"), 'r') as f:
                inputs_raw = [line.strip() for line in f]
                smiles_list = inputs_raw[:]
            # read in targets if avaliable
            target_path = os.path.join(args.data_dir, f"{base}.target")
            if os.path.exists(target_path):
                with open(target_path, "r") as f:
                    targets_raw = [line.rstrip("\n") for line in f]
            else:
                targets_raw = None

        # Canonicalize SMILES if needed
        smiles_df = pd.DataFrame(smiles_list, columns=["source"])
        if smiles_df['source'].str.contains("Prot:|Deprot:").any():
            smiles_df[['prefix', 'just_smiles']] = smiles_df['source'].str.split(':', expand=True)
            canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(m), canonical=True) for m in smiles_df['just_smiles']]
            smiles_df['canonical'] = canonical
            smiles_df['combined'] = smiles_df['prefix'] + ':' + smiles_df['canonical']
            smiles_list = smiles_df['combined'].tolist()

        # Create dataset and dataloader
        testset = TaskPrefixDataset(
            tokenizer,
            data_dir=args.data_dir,
            type_path=base,
            prefix=args.prefix or task.prefix,
            max_source_length=task.max_source_length,
            max_target_length=task.max_target_length,
            separate_vocab=(task.output_layer != 'seq2seq'),
        )
        test_loader = DataLoader(testset,
        batch_size=args.batch_size,
        collate_fn=partial(data_collator, pad_token_id=tokenizer.pad_token_id),
        )

        # load in T5ForConditionalGeneration used in seq2seq model 
        model_cls = T5ForConditionalGeneration
        #allow possibility to predict using best checkpoint
        if args.best_cp:
            model = model_cls.from_pretrained(best_files[0])
        else:
            model = model_cls.from_pretrained(args.model_dir)
        
        model = model.to(device)
        model.eval()

        # Prediction
        if task.output_layer == 'seq2seq':
            predictions = [[] for _ in range(args.num_preds)]
            task_params = {
                "early_stopping": True,
                "max_length": task.max_target_length,
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_preds,
                "decoder_start_token_id": tokenizer.pad_token_id,
            }
            for batch in tqdm(test_loader, desc="Predicting"):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                del batch['labels']
                with torch.no_grad():
                    output = model.generate(**batch, **task_params)
                for i, pred in enumerate(output):
                    decoded = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces = False)
                    predictions[i % args.num_preds].append(standize(decoded))

            pred_df = pd.DataFrame({f'prediction_{i+1}': preds for i, preds in enumerate(predictions)})
            # save prediction .csv file with inputs, targets, and predictions
            out = pd.DataFrame({"input": smiles_list})
            if targets_raw is not None:
                out["target"] = targets_raw
            test_df = pd.concat([out, pred_df], axis=1)

        # compute metrics for seq2seq task
        if targets_raw is not None:
            for i, preds in enumerate(predictions):
                test_df['prediction_{}'.format(i + 1)] = preds
                test_df['prediction_{}'.format(i + 1)] = test_df['prediction_{}'.format(i + 1)].apply(standize)
            test_df['rank'] = test_df.apply(lambda row: get_rank(row, 'prediction_', args.num_preds), axis=1)

            correct = 0
            invalid_smiles = 0
            for i in range(1, args.num_preds+1):
                correct += (test_df['rank'] == i).sum()
                invalid_smiles += (test_df['prediction_{}'.format(i)] == '').sum()
                print('Top-{}: {:.1f}% || Invalid {:.2f}%'.format(i, correct/len(test_df)*100, \
                    invalid_smiles/len(test_df)/i*100))
        if args.regression_targets is not None:
            reg_targets = pd.read_csv(args.regression_targets, names=['target'])
            test_df["target"] = pd.to_numeric(reg_targets["target"], errors="coerce")
            new_test_df = test_df.copy()
        else:
            new_test_df = test_df.copy()
        
        new_test_df[['prefix','smiles']] = new_test_df['input'].str.split(':',expand=True)
        pairs, skip_indices = [], []
        for i, (x, y, z) in enumerate(zip(new_test_df['smiles'], new_test_df['prediction_1'], new_test_df['prediction_2'])):
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
        new_test_df = new_test_df.drop(skip_indices).reset_index(drop=True)
        print(f"Skipped {len(skip_indices)} rows due to missing predictions or invalid charge transitions.")

        # add pairs into dataset to be predicted by regression model
        new_test_df['source'] = pairs

        # preparing to enter regression ensemble model loop
        if not args.regression_model_dir:
            raise ValueError("When --unify True, you must provide --regression_model_dir")
        reg_model_dir = args.regression_model_dir
        reg_smiles_list = new_test_df['source'].astype(str).tolist()
        reg_targets_list = new_test_df['target'].astype(float).tolist()
    else:
        reg_model_dir = args.model_dir 
        reg_smiles_list = None
        reg_targets_raw = None
    all_predictions, all_targets = [], []
    for i in range(1,11):
        # option to chose the best checkpoint to run prediction (read in config)
        if args.best_cp == True:
            best_files = glob.glob(reg_model_dir + str(i) +'/best_*/')
            config = T5Config.from_pretrained(best_files[0])
        else:
            config = T5Config.from_pretrained(reg_model_dir + str(i) )
        
        # get task type information
        task = T5ChemTasks[config.task_type]

        # get the type of Tokenizer needed 
        tokenizer_type = getattr(config, "tokenizer")
        tokenizer_map = {"simple": SimpleTokenizer,"atom": AtomTokenizer, "selfies": SelfiesTokenizer }
        Tokenizer = tokenizer_map.get(tokenizer_type)
        tokenizer = Tokenizer(vocab_file=os.path.join(reg_model_dir, str(i)+'/vocab.pt'))
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
        if args.unify:
            testset = TaskPrefixDataset(
                        tokenizer,
                        smiles_list=reg_smiles_list,
                        prefix=args.prefix or task.prefix,
                        max_source_length=task.max_source_length,
                        max_target_length=task.max_target_length,
                        separate_vocab=(task.output_layer != 'seq2seq'),
                        )
        else:
            testset = TaskPrefixDataset(
                        tokenizer,
                        data_dir=args.data_dir,
                        type_path=base,
                        prefix=args.prefix or task.prefix,
                        max_source_length=task.max_source_length,
                        max_target_length=task.max_target_length,
                        separate_vocab=(task.output_layer != 'seq2seq'),
                        )

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
            best_files = glob.glob(reg_model_dir + str(i) +'/best_*/')
            model = T5ForProperty.from_pretrained(best_files[0])
        else:
            model = T5ForProperty.from_pretrained(reg_model_dir + str(i))
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
    if args.unify == True:
        d = {'targets': reg_targets_list}
    else:
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
        args.prediction = os.path.join(reg_model_dir, 'predictions_'+base+'.csv')
    else: 
        df.to_csv(args.prediction, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    predict(args)
