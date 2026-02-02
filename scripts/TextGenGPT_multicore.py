
#Imports
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import json
import sys
import os
from pprint import pprint
import multiprocessing as mp
from tqdm import tqdm
import OpenAI_lib as ol

client = ol.get_client_local()


#Define wished output structure and method for prompting GPT
class GeneratedTexts(BaseModel):
    alku: str
    teksti: str

def promptGPT(model, user_prompt, reasoning_effort, format):
    completion = client.responses.parse(
            model=model,
            input=user_prompt,
            reasoning={ "effort": reasoning_effort},
            text_format=format,
        )
    return completion.output_parsed

def promptTask(d, model_name, effort):
    base_prompt = "Tehtäväsi on kirjoittaa annetun tekstin "
    base_prompt += d['first_line_type']+" perusteella kokonainen "
    base_prompt += d['register']+", joka on "+str(d['text_sent_amount'])+" virkkeen pituinen.\n"
    base_prompt += "Annettu teksti: "+d['first_line']
    test = promptGPT(model_name, base_prompt, effort, GeneratedTexts)
    return {'id':d['id'], 'model':model_name, 'effort':effort, 'register':d['register'], 'first_line_type':d['first_line_type'], 'first_line':test.alku, 'text':test.teksti, 'text_sent_amount':d['text_sent_amount']}

def main(cmd_args):

    model_name = cmd_args[0]
    effort = cmd_args[1]
    output_name = cmd_args[2]
    
    tdt_doc_data = []
    with open("data/output/TDT_doc_data.jsonl") as reader:
        for l in reader:
            tdt_doc_data.append(json.loads(l))
    gotten_results = []
    if os.path.exists("data/output/"+output_name+".jsonl"):
        with open("data/output/"+output_name+".jsonl", 'r') as reader:
            for l in reader:
                if len(l) > 0:
                    gotten_results.append(json.loads(l)['id'])

    #results = []
    with tqdm(total=len(tdt_doc_data)) as pbar:
        for d in tdt_doc_data:
            if d['id'] not in gotten_results:
                res = promptTask(d, model_name, effort)
                with open("data/output/"+output_name+".jsonl", 'a') as writer:
                    writer.write(json.dumps(res)+'\n')
                    pbar.update()
            else:
                pbar.update()

    
    #with open("data/output/"+output_name+".jsonl", 'w') as writer:
    #    for f in results:
    #        writer.write(json.dumps(f)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])