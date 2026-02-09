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
    if regen_ids_texts[0].get('custom_id'):
        regen_texts = {x['custom_id'][x['custom_id'].find(".jsonl_")+7:]:x['text'] for x in regen_ids_texts}
    else:
        regen_texts = {str(i):regen_ids_texts[i]['text'] for i in range(len(regen_ids_texts))}

    #prompt = "Tehtäväsi on arvioida suomenkielisten tekstien laatua ja valita kahdesta annetusta tekstistä se, kumpi on laadukkaampi. Vastaa ainoastaan Teksti 1, jos ensimmäinen teksti on parempi, ja Teksti 2, jos toinen teksit on parempi.\n"
    for i, x in enumerate(human_texts):

        prompt = "Tehtäväsi on arvioida suomenkielisten tekstien laatua ja valita kahdesta annetusta tekstistä se, kumpi on laadukkaampi. Vastaa ainoastaan Teksti 1, jos ensimmäinen teksti on parempi, ja Teksti 2, jos toinen teksit on parempi.\n"
        
        prompt = """ Tehtäväsi on arvioida suomenkielisten tekstien laatua ja valita kahdesta annetusta tekstistä se, kumpi on laadukkaampi.
        Valitse joko Teksti 1 tai Teksti 2, riippuen siitä, kumpi on laadukkaampi.
        Sinun täytyy valita joko Teksti 1 tai Teksti 2.

        Alla on annettu arviointikriteereitä, joita sinun tulee noudattaa:
        (1) Teksti on laadukkaampi, kun siinä on vähemmän oikeinkirjoitusvirheitä.
        (2) Teksti on laadukkaampi, jos siinä on taivutettu sanoja kielioppisääntöjen mukaisesti.
        (3) Teksti on laadukkaampi, jos se on koherentti.
        (4) Tekstin pituudella ei ole merkitystä sen laadun kannalta.
        (5) Tekstien annetulla järjestyksellä ei ole merkitystä, vaan on yhtä todennäköistä, että Teksti 1 tai Teksti 2 on laadukkaampi.

        Vastaa vain ja ainoastaan seuraavalla tavalla:

        **Vastaus:** <Teksti 1 tai Teksti 2>

        Arvioitavat tekstit:
        ```


"""
        #50/50 chance whether human or regen text is the first choice
        if random.randint(0,1) == 0:
            prompt += "\nTeksti 1: "+x+'```\n'
            prompt += "```\nTeksti 2: "+regen_texts[str(i)]+'\n```'
            dpo_dataset.append({"prompt":prompt, "chosen":"Teksti 1", "rejected":"Teksti 2"})
        else:
            prompt += "\nTeksti 1: "+regen_texts[str(i)]+'```\n'
            prompt += "```\nTeksti 2: "+x+'\n```'
            dpo_dataset.append({"prompt":prompt, "chosen":"Teksti 2", "rejected":"Teksti 1"})

    with open("data/DPO_datasets/"+input_human_dataset[:-6]+"_dpo.jsonl", 'w') as writer:
        for d in dpo_dataset:
            writer.write(json.dumps(d)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])