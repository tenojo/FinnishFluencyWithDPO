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
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import DatasetDict, Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="The pre-trained model from Hugging Face to use as basis: "
        "https://huggingface.co/models"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--dp",
        type=str,
        default="data/DPO_datasets/TDT_doc_data_dpo.jsonl",
        help="Evaluation dataset path",
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

    # #### Loading the model
    # Let's start with getting the appropriate tokenizer.
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    # Load the actual base model from Hugging Face
    if rank == 0:
        print("Loading model and tokenizer")

    quantization_config = None
    if args.bnb_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        quantization_config = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        )

    #model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    

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
    

    ds_items = []
    with open(args.dp, 'r') as reader:
        for l in reader:
            if len(l) > 0:
                ds_items.append(json.loads(l.strip()))
    print("Loaded samples!")

    ds = Dataset.from_list(ds_items)
    ds_items = []
    del ds_items
    print("Dataset created!")
    for x in ds:
        print(model(**tokenizer(x['prompt'], return_tensors="pt")))