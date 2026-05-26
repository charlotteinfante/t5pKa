import argparse
import logging
import os
import glob
import random
import shutil
from functools import partial
from typing import Dict

import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from transformers import DataCollatorForLanguageModeling, T5Config, T5ForConditionalGeneration

try:
    from .compat import make_t5_config_kwargs, make_training_arguments, preprocess_logits_for_metrics
    from .data_utils import (
        AccuracyMetrics,
        CalMSELoss,
        F1_AUCMetrics,
        T5ChemTasks,
        TaskPrefixDataset,
        TaskSettings,
        data_collator,
    )
    from .model import T5ForProperty
    from .mol_tokenizers import AtomTokenizer, MolTokenizer, SelfiesTokenizer, SimpleTokenizer
    from .trainer import EarlyStopTrainer
except ImportError:
    from compat import make_t5_config_kwargs, make_training_arguments, preprocess_logits_for_metrics
    from data_utils import (
        AccuracyMetrics,
        CalMSELoss,
        F1_AUCMetrics,
        T5ChemTasks,
        TaskPrefixDataset,
        TaskSettings,
        data_collator,
    )
    from model import T5ForProperty
    from mol_tokenizers import AtomTokenizer, MolTokenizer, SelfiesTokenizer, SimpleTokenizer
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
        help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type to use. ('product', 'reactants', 'reagents', \
            'regression', 'classification', 'pretrain', 'mixed')",
    )
    parser.add_argument(
        "--pretrain",
        default='',
        help="Path to a pretrained model. If not given, we will train from scratch",
    )
    parser.add_argument(
        "--pretrain_best_cp",
        default=False,
        type=bool,
        help="use the best step based on lowest loss for pretrained model trained using t5chem; default is set to False",
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
        "--random_seed",
        default=8570,
        type=int,
        help="The random seed for model initialization",
    )
    parser.add_argument(
        "--num_epoch",
        default=100,
        type=int,
        help="Number of epochs for training.",
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
        "--num_workers",
        default=0,
        type=int,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--init_lr",
        default=5e-4,
        type=float,
        help="The initial leanring rate for model training",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help="The number of classes in classification or regression task.",
    )
    parser.add_argument(
        "--canonicalize_smiles",
        action="store_true",
        help="Canonicalize SMILES/reaction strings with the active RDKit version before tokenization.",
    )


def train(args):
    print(args)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # In multi-GPU runs this keeps datasets read in the same order.
    random.seed(args.random_seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    assert args.task_type in T5ChemTasks, \
        "only {} are currenly supported, but got {}".\
            format(tuple(T5ChemTasks.keys()), args.task_type)
    task: TaskSettings = T5ChemTasks[args.task_type]

    if args.task_type in ['regression','macropka','micropka']:
        if os.path.isfile(os.path.join(args.data_dir,'MinMaxScaler.gz')):
            scaler = joblib.load(os.path.join(args.data_dir,'MinMaxScaler.gz'))
        else:
            all_targets = np.loadtxt(os.path.join(args.data_dir,'train.target'),delimiter=',')
            if len(all_targets.shape) == 1:
                all_targets = all_targets.reshape(-1, 1)
            scaler = MinMaxScaler(clip=True)
            scaler.fit(all_targets)
            joblib.dump(scaler, os.path.join(args.data_dir,'MinMaxScaler.gz'))
        args.num_classes = scaler.n_features_in_
    else:
        scaler = None

    if args.pretrain: # retrieve information from pretrained model
        if task.output_layer == 'seq2seq':
            model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
        else:
            model = T5ForProperty.from_pretrained(
                args.pretrain,
            )   # use model pretrained setting for now, resize later if inconsistent with our dataset
            model.head_type = task.output_layer
            model.config.head_type = model.head_type
            if args.num_classes and (args.num_classes != getattr(model.config, "num_classes", None)):
                model.resize_num_classes(args.num_classes)
            
        if not hasattr(model.config, 'tokenizer'):
            logging.warning("No tokenizer type detected, will use SimpleTokenizer as default")
        tokenizer_type = getattr(model.config, "tokenizer", 'simple')
        if args.pretrain_best_cp == True:
            pretrain_directory =  glob.glob(args.pretrain)[0]
            vocab_path = os.path.join(pretrain_directory, '..', 'vocab.pt')
        else: 
            vocab_path = os.path.join(args.pretrain, 'vocab.pt')
        if not os.path.isfile(vocab_path):
            vocab_path = args.vocab
            if not vocab_path:
                raise ValueError(
                        "Can't find a vocabulary file at path '{}'.".format(args.pretrain)
                    )
        tokenizer = tokenizer_map[tokenizer_type](vocab_file=vocab_path, task_prefix=["Pairs:","Deprot:","Prot:"], max_size=116) #changed
        model.config.tokenizer = tokenizer_type # type: ignore
        model.config.task_type = args.task_type # type: ignore
    else:
        if not args.tokenizer:
            warn_msg = "This model is trained from scratch, but no \
                tokenizer type is specified, will use simple tokenizer \
                as default for this training."
            logging.warning(warn_msg)
            args.tokenizer = 'simple'
        assert args.tokenizer in ('simple', 'atom', 'selfies'), \
            "{} tokenizer is not supported."

        # if path to vocab file given, then use it
        if args.vocab == '':    #added 
            vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/'+args.tokenizer+'.pt') #added 
        else:
            vocab_path = args.vocab #added
            
        tokenizer = tokenizer_map[args.tokenizer](vocab_file=vocab_path)
        #tokenizer.add_tokens(["Pairs:", "acidic:", "basic:"])
        config = T5Config(**make_t5_config_kwargs(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_layers=4,
            num_heads=8,
            d_model=256,
            tokenizer=args.tokenizer,
            task_type=args.task_type,
        ))
        if task.output_layer == 'seq2seq':
            model = T5ForConditionalGeneration(config)
        else:
            model = T5ForProperty(config, head_type=task.output_layer, num_classes=args.num_classes)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))
    if scaler is not None:
        shutil.copyfile(
            os.path.join(args.data_dir, 'MinMaxScaler.gz'),
            os.path.join(args.output_dir, 'MinMaxScaler.gz'),
        )
    dataset = TaskPrefixDataset(
        tokenizer, 
        data_dir=args.data_dir,
        prefix=task.prefix,
        max_source_length=task.max_source_length,
        max_target_length=task.max_target_length,
        separate_vocab=(task.output_layer != 'seq2seq'),
        canonicalize_smiles=args.canonicalize_smiles,
        type_path="train",
    )
    data_collator_padded = partial(
        data_collator, pad_token_id=tokenizer.pad_token_id, normalize=scaler)

    do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
    if do_eval:
        eval_strategy = "steps"
        eval_iter = TaskPrefixDataset(
            tokenizer, 
            data_dir=args.data_dir,
            prefix=task.prefix,
            max_source_length=task.max_source_length,
            max_target_length=task.max_target_length,
            separate_vocab=(task.output_layer != 'seq2seq'),
            canonicalize_smiles=args.canonicalize_smiles,
            type_path="val",
        )
    else:
        eval_strategy = "no"
        eval_iter = None

    if task.output_layer == 'seq2seq':
        compute_metrics = AccuracyMetrics
    elif task.output_layer == 'regression':
        compute_metrics = partial(CalMSELoss, scaler=scaler)
    else: # Classification
        compute_metrics = F1_AUCMetrics

    training_args = make_training_arguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        eval_strategy=eval_strategy,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.log_step,
        per_device_eval_batch_size=args.batch_size,
        save_steps=10000,
        save_total_limit=5,
        learning_rate=args.init_lr,
        prediction_loss_only=(compute_metrics is None),
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = EarlyStopTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    print(args)
    print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    train(args)
