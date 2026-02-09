import os
import json
import sys
import random

def main(cmd_args):
    input_dataset_name = cmd_args[0]

    dpo_dataset = []
    with open("data/DPO_datasets/"+input_dataset_name+".jsonl", 'r') as reader:
        for l in reader:
            if len(l) > 1:
                dpo_dataset.append(json.loads(l.strip()))
    
    pre_train_ds = []
    for x in dpo_dataset:
        pre_train_ds.append({'prompt':x['prompt'], 'completion':x['chosen']})

    with open("data/Pretrain_datasets/"+input_dataset_name+".jsonl", 'w') as writer:
        for d in pre_train_ds:
            writer.write(json.dumps(d)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])