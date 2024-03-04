import argparse
import logging
import os
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
import h5py
import joblib
from sklearn.preprocessing import MinMaxScaler
from transformers import (DataCollatorForLanguageModeling, T5Config,
                          T5ForConditionalGeneration, TrainingArguments,
                          EarlyStoppingCallback)

from data_utils import (AccuracyMetrics, CalMSELoss, LineByLineTextDataset,
                        T5ChemTasks, TaskPrefixDataset, TaskSettings,
                        PropertyPretrainDataset, data_collator)
from model import T5ForProperty
from mol_tokenizers import (AtomTokenizer, MolTokenizer, SelfiesTokenizer,
                            SimpleTokenizer)
from trainer import EarlyStopTrainer

tokenizer_map: Dict[str, MolTokenizer] = {
    'simple': SimpleTokenizer,  # type: ignore
    'atom': AtomTokenizer,  # type: ignore
    'selfies': SelfiesTokenizer,    # type: ignore
}


def add_args(parser):
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        nargs='*',
        help="The input data dir(s). Should at least contain train.source, train.hdf5 (if it's regression) or train.txt (text-filling)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pretrain",
        default='',
        help="Path to a pretrained model. If not given, we will train from scratch",
    )
    parser.add_argument(
        "--vocab",
        default='',
        help="Vocabulary file to load.",
    )
    parser.add_argument(
        "--tokenizer",
        default='',
        help="Tokenizer to use. ('simple', 'atom', 'selfies')",
    )
    parser.add_argument(
        "--task_type",
        default='',
        help="Task Type ('regression', 'seq2seq'), default: auto-detect",
    )
    parser.add_argument(
        "--random_seed",
        default=8570,
        type=int,
        help="The random seed for model initialization",
    )
    parser.add_argument(
        "--save_total_limit",
        default=5,
        type=int,
        help="The number of checkpoints to save during pretraining",
    )
    parser.add_argument(
        "--log_step",
        default=5000,
        type=int,
        help="Logging after every log_step",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--init_lr",
        default=5e-4,
        type=float,
        help="The initial leanring rate for model training",
    )


def train(args):
    print(args)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # this one is needed for torchtext random call (shuffled iterator)
    # in multi gpu it ensures datasets are read in the same order
    random.seed(args.random_seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    task_type = 'regression' if os.path.isfile(os.path.join(args.data_dir[0],'train.hdf5')) else 'seq2seq'
    task_type = args.task_type or task_type
    if args.pretrain: # retrieve information from pretrained model
        if task_type == 'seq2seq': # sequence-to-sequence mask-filling pretraining
            model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
        else:   # property pretraining
            # if training on preperties, check dimension
            h5file = h5py.File(os.path.join(args.data_dir[0], "train.hdf5"), "r")
            dim = h5file['dataset'].shape[-1]
            model = T5ForProperty.from_pretrained(
                args.pretrain, 
                head_type = "regression",
                num_classes = dim,
            )
        if not hasattr(model.config, 'tokenizer'):
            logging.warning("No tokenizer type detected, will use SimpleTokenizer as default")
        tokenizer_type = getattr(model.config, "tokenizer", 'simple')
        vocab_path = os.path.join(args.pretrain, 'vocab.pt')
        if not os.path.isfile(vocab_path):
            vocab_path = args.vocab
            if not vocab_path:
                raise ValueError(
                        "Can't find a vocabulary file at path '{}'.".format(args.pretrain)
                    )
        tokenizer = tokenizer_map[tokenizer_type](vocab_file=vocab_path)
        model.config.tokenizer = tokenizer_type # type: ignore
        model.config.task_type = "pretrain" # type: ignore
    else:
        if not args.tokenizer:
            warn_msg = "This model is trained from scratch, but no \
                tokenizer type is specified, will use simple tokenizer \
                as default for this training."
            logging.warning(warn_msg)
            args.tokenizer = 'simple'
        assert args.tokenizer in ('simple', 'atom', 'selfies'), \
            "{} tokenizer is not supported."
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/'+args.tokenizer+'.pt')
        tokenizer = tokenizer_map[args.tokenizer](vocab_file=vocab_path)
        config = T5Config(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_past=True,
            num_layers=4,
            num_heads=8,
            d_model=256,
            tokenizer=args.tokenizer,
            task_type='pretrain',
        )
        if task_type == 'seq2seq': # sequence-to-sequence mask-filling pretraining
            model = T5ForConditionalGeneration(config)
        else:
            h5file = h5py.File(os.path.join(args.data_dir[0], "train.hdf5"), "r")
            dim = h5file['dataset'].shape[-1]
            model = T5ForProperty(config, head_type='regression', num_classes=dim)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))
    for i,folder in enumerate(args.data_dir):
        if task_type == 'seq2seq': # sequence-to-sequence mask-filling pretraining
            dataset = LineByLineTextDataset(
                tokenizer=tokenizer, 
                file_path=os.path.join(folder,'train.source'),
                block_size=400,
                prefix='Fill-Mask:',
            )
            data_collator_padded = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
            do_eval = os.path.exists(os.path.join(folder, 'val.source'))
            if do_eval:
                eval_strategy = "steps"
                eval_iter = LineByLineTextDataset(
                    tokenizer=tokenizer, 
                    file_path=os.path.join(folder,'val.source'),
                    block_size=400,
                    prefix='Fill-Mask:',
                )
            else:
                eval_strategy = "no"
                eval_iter = None
            # compute_metrics = None 
        else:
            dataset = PropertyPretrainDataset(
                tokenizer, 
                data_dir=folder,
                prefix='regression',
                max_source_length=400,
                type_path="train",
            )
            data_collator_padded = partial(
                data_collator, pad_token_id=tokenizer.pad_token_id, normalize=None) # should be pre-scaled for faster speed
            do_eval = os.path.exists(os.path.join(folder, 'val.hdf5'))
            if do_eval:
                eval_strategy = "steps"
                eval_iter = PropertyPretrainDataset(
                    tokenizer, 
                    data_dir=folder,
                    prefix='regression',
                    max_source_length=400,
                    type_path="val",
                )
            else:
                eval_strategy = "no"
                eval_iter = None
            # compute_metrics = CalMSELoss

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            evaluation_strategy=eval_strategy,
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            logging_steps=1000,
            per_device_eval_batch_size=args.batch_size,
            save_steps=1000,
            save_total_limit=1,
            disable_tqdm=True,
            learning_rate=args.init_lr,
            prediction_loss_only=True,
            load_best_model_at_end=True,
        )

        trainer = EarlyStopTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator_padded,
            train_dataset=dataset,
            eval_dataset=eval_iter,
            compute_metrics=None,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        print("logging dir: {}".format(training_args.logging_dir))
        print(folder, 'trained.')
        round_to_save = len(args.data_dir)//args.save_total_limit or len(args.data_dir) # in case = 0
        if (i+1) % round_to_save == 0:
            trainer.save_model(os.path.join(args.output_dir, str(i+1)))
    trainer.save_model(args.output_dir) # always save the last checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    train(args)
