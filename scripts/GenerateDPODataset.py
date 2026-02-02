import os
import json
import sys
import random

def main(cmd_args):
    input_human_dataset = cmd_args[0]
    input_regen_dataset = cmd_args[1]

    dpo_dataset = []

    human_texts = []
    regen_ids_texts = []
    with open("data/HumanDatasets/"+input_human_dataset, 'r') as reader:
        for l in reader:
            if len(l) > 1:
                human_texts.append(json.loads(l.strip())['text'])
    with open("data/GPTRegens/"+input_regen_dataset, 'r') as reader:
        for l in reader:
            if len(l) > 1:
                regen_ids_texts.append(json.loads(l.strip()))
    regen_texts = {x['custom_id'][x['custom_id'].find(".jsonl_")+7:]:x['text'] for x in regen_ids_texts}

    #prompt = "Tehtäväsi on arvioida suomenkielisten tekstien laatua ja valita kahdesta annetusta tekstistä se, kumpi on laadukkaampi. Vastaa ainoastaan Teksti 1, jos ensimmäinen teksti on parempi, ja Teksti 2, jos toinen teksit on parempi.\n"
    for i, x in enumerate(human_texts):
        prompt = "Tehtäväsi on arvioida suomenkielisten tekstien laatua ja valita kahdesta annetusta tekstistä se, kumpi on laadukkaampi. Vastaa ainoastaan Teksti 1, jos ensimmäinen teksti on parempi, ja Teksti 2, jos toinen teksit on parempi.\n"
        #50/50 chance whether human or regen text is the first choice
        if random.randint(0,1) == 0:
            prompt += "Teksti 1: "+x+'\n'
            prompt += "Teksti 2: "+regen_texts[str(i)]+'\n'
            dpo_dataset.append({"prompt":prompt, "accepted":"Teksti 1", "rejected":"Teksti 2"})
        else:
            prompt += "Teksti 1: "+regen_texts[str(i)]+'\n'
            prompt += "Teksti 2: "+x+'\n'
            dpo_dataset.append({"prompt":prompt, "accepted":"Teksti 2", "rejected":"Teksti 1"})

    with open("data/DPO_datasets/"+input_human_dataset[:-6]+"_dpo.jsonl", 'w') as writer:
        for d in dpo_dataset:
            writer.write(json.dumps(d)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])