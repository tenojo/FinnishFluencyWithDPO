#!/usr/bin/env python
# coding: utf-8

# Simple benchmarking script that does fine-tuning on a given Hugging
# Face model with IMDB movie reviews
#
# Adapted from the exercises of the LUMI AI workshop course:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop

import argparse
import os
import sys
import time
import json
import torch
from sklearn import metrics
import numpy as np
from datasets import DatasetDict, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer,
                          TrainingArguments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="TurkuNLP/bert-base-finnish-cased-v1",
        help="The pre-trained model from Hugging Face to use as basis: "
        "https://huggingface.co/models"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="The number of CPU worker processes to use.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="If set, continue from a previously interrupted run. "
        "Otherwise, overwrite existing checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="The number of training steps.",
    )
    parser.add_argument(
        "--4bit",
        dest="bnb_4bit",
        action='store_true',
        help="Use 4bit quantization with bitsandbytes: "
        "https://huggingface.co/docs/bitsandbytes/main/en/index"
    )
    args, _ = parser.parse_known_args()
    # Some constants
    os.environ['WANDB_MODE'] = 'disabled'
    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Then we determine the device on which to train the model.
    if rank == 0:
        print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        print(f"Using GPU {local_rank}, device name: {torch.cuda.get_device_name(device)}")
    else:
        print(f"No GPU found, using CPU instead. (Rank: {local_rank})")
        device = torch.device("cpu")

    if rank == 0 and args.batch_size % world_size != 0:
        print(f"ERROR: batch_size={args.batch_size} has to be a multiple of "
              f"the number of GPUs={world_size}!")
        sys.exit(1)

    # We also ensure that output paths exist
    model_name = args.model.replace('/', '_')

    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, model_name)

    # #### Loading the model
    # Let's start with getting the appropriate tokenizer.
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    # Load the actual base model from Hugging Face
    if rank == 0:
        print("Loading model and tokenizer")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        )

    #model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.resume,
        # save_strategy="no",  # good for testing
        save_strategy="steps",   # use these if you actually want to save the model
        save_steps=1000,
        save_total_limit=4,
        eval_strategy="steps",
        eval_steps=500,  # compute validation loss every 200 steps
        learning_rate=1e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=train_batch_size // world_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        # report_to=["tensorboard"],  # log statistics for tensorboard
        ddp_find_unused_parameters=False,
    )

    # #### Setting up preprocessing of training data

    # IMDb examples are presented as a dictionary:
    # {
    #    'text': the review text as a string,
    #    'label': a sentiment label as an integer,
    # }.
    # We tokenize the text and add the special token for indicating the end of the
    # text at the end of each review. We also truncate reviews to a maximum
    # length to avoid excessively long sequences during training.
    # As we have no use for the label, we discard it.
    def tokenize(ex):
            return tokenizer(
                ex['text'],
                max_length=512,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
            )
    

    corr_samples = []
    with open('data/output/TDT_corrupted_shuffle_forms_1.jsonl', 'r') as reader:
        for l in reader:
            if len(l) > 0:
                corr_samples.append(json.loads(l))
    print("Loaded corrupt samples!")

    clean_samples = []
    with open('data/output/TDT_doc_data.jsonl', 'r') as reader:
        for l in reader:
            if len(l) > 0:
                clean_samples.append(json.loads(l))
    print("Loaded clean samples!")

    #Make clean and corrupted samples compatible
    for i, x in enumerate(clean_samples):
        t = {}
        t['id'] = x['id']
        t['text'] = x['text']
        t['score'] = float(1.0)
        t['corruptions'] = []
        clean_samples[i] = t

    for i, x in enumerate(corr_samples):
        t = x
        t['score'] = float(x['bleurt_score'])
        corr_samples[i]=t
    print("Finished formatting!")
    ds = Dataset.from_list(clean_samples+corr_samples).rename_column("score", "label").train_test_split(test_size=0.2)
    corr_samples = []
    del corr_samples
    clean_samples = []
    del clean_samples
    print("Dataset created!")
    # %%
    def tokenize(ex):
        return tokenizer(
            ex['text'],
            max_length=512,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
    print("Moving to tokenization!")
    ds = ds.map(tokenize, num_proc=training_args.dataloader_num_workers, batch_size=8, batched=True).select_columns(['input_ids', 'attention_mask', 'label'])
    for x in ds:
        ds[x].set_format("pt", columns=["input_ids"], output_all_columns=True) 
    tok_train = ds['train']
    # We split a small amount of training data as "validation" test
    # set to keep track of evaluation of the loss on non-training data
    # during training.  This is purely because computing the loss on
    # the full evaluation dataset takes much longer.
    tok_val = ds['test'].train_test_split(test_size=0.1, keep_in_memory=True)['test']

    # Metrics for the model
    def compute_metrics_for_regression(eval_pred):
        logits, labels = eval_pred
        labels = labels.reshape(-1, 1)

        mse = metrics.mean_squared_error(labels, logits)
        rmse = metrics.root_mean_squared_error(labels, logits)
        mae = metrics.mean_absolute_error(labels, logits)
        r2 = metrics.r2_score(labels, logits)
        smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape, 'accuracy':mse}

    collator = data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics_for_regression,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir)
    if rank == 0:
        print()
        print("Training done, you can find the final model (and checkpoints) in", output_dir)