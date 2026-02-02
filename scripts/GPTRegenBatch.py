
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
import time
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import Response
from openai.lib._parsing._responses import parse_response

client = ol.get_client_local()


#Define wished output structure and method for prompting GPT
class GeneratedTexts(BaseModel):
    teksti: str

def promptGPT(model, user_prompt, reasoning_effort, format):
    completion = client.responses.parse(
            model=model,
            input=user_prompt,
            reasoning={ "effort": reasoning_effort},
            text_format=format,
        )
    return completion.output_parsed

def generateBatchItem(og_dataset_name, batch_id, og_text, model_name, effort, text_format):
    bitem = {"custom_id":f"{og_dataset_name}_{batch_id}", "method":"POST", "url":"/v1/responses"}
    base_prompt = "Muokkaa annetusta tekstistä kömpelömpi versio. Pidä huolta siitä että teksti pysyy luettavana. Alla on annettu esimerkki, jossa on ensin annettu teksti ja tämän jälkeen siitä muokattu kömpelömpi versio.\nEsimerkki annetusta tekstistä: Menimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Rakensimme sen avulla pienen sillan puron yli. Jätimme laudan metsään sellaiseen paikkaan, jonka varmasti muistamme seuraavalla retkellä.\nEsimerkki kömpelöstä versiosta: Eilen menimme luokan kanssa retkelle, ja ensimmäinen paikka oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan. Rakensimme laudan avulla pienen sillan puron yli, ja se jäi metsään paikalle, jonka muistamme varmasti seuraavalla retkellä.\nAnnettu teksti:"
    base_prompt += og_text.replace('\n', '')
    body = {
        "model":model_name,
        "input":base_prompt,
        "reasoning":{"effort":effort},
        "text":{"format":text_format},
        }
    bitem["body"] = body
    return bitem

def readBatchResponseContents(lines):
    completions = []
    #Slightly complicated way of getting around some output format errors and still getting all response texts
    #Should work always, but may need to change in the future
    for line in lines:
        infor = json.loads(line)
        cust_id = infor['custom_id']
        tex = infor['response']['body']['output']
        temp = ""
        for t in tex:
            has_text = t.get('content', None)
            if has_text:
                if isinstance(has_text, list):
                    has_text = has_text[0]
                has_text = has_text['text']
                if isinstance(has_text, str):
                    temp += has_text[11:-2]
                else:
                    try:
                        temp += has_text['text'].get('teksti', '')
                    except:
                        pprint(has_text)
        completions.append({"custom_id":cust_id, "text":temp})
    return completions


def main(cmd_args):

    model_name = cmd_args[0]
    effort = cmd_args[1]
    human_dataset_name = cmd_args[2]
    output_name = cmd_args[3]
    if len(cmd_args) > 4:
        existing_batch_job_id = cmd_args[4]

    response_format = type_to_text_format_param(GeneratedTexts)

    #If no cache file exists
    if not os.path.exists("data/RegenerationCache/"+output_name+".jsonl"):
        batch_items = []
        with open("data/HumanDatasets/"+human_dataset_name, 'r') as reader:
            for i,l in enumerate(reader):
                og_text = json.loads(l)['text']
                batch_items.append(generateBatchItem(human_dataset_name, i, og_text, model_name, effort, response_format))

        
        with open("data/RegenerationCache/"+output_name+".jsonl", 'w') as writer:
            for f in batch_items:
                writer.write(json.dumps(f)+'\n')

    #If we want to submit a batch job with the existing cache file (usually the remaining, uncompleted items)
    if not len(cmd_args)>4:
        batch_input_file = client.files.create(
            file=open("data/RegenerationCache/"+output_name+".jsonl", "rb"),
            purpose="batch"
        )

        #print(batch_input_file)
        print("Batch file created! Proceeding to send it to Open AI API")

        batch_input_file_id = batch_input_file.id
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": "batch processing job"
            }
        )

        batch_job = client.batches.retrieve(batch_job.id)
    #If we want to process some batch job that already completed
    else:
        batch_job = client.batches.retrieve(existing_batch_job_id)
    
    print("Initial batch job information:\n")
    print(batch_job)
    ss = ['validating', 'in_progress', 'finalizing', 'cancelling']
    print("\n\nWaiting for response...\n\n")
    while client.batches.retrieve(batch_job.id).status in ss:
        time.sleep(300)
    print("Got a response from the API:\n")
    batch_job = client.batches.retrieve(batch_job.id)
    print(batch_job)
    if batch_job.error_file_id:
        print('\n\n\n')
        #print(client.files.content(batch_job.error_file_id).content)
        #Get all successfully processed items in the batch
        if batch_job.output_file_id:
            result_file_id = batch_job.output_file_id
            file_response = client.files.content(result_file_id)

            completions = readBatchResponseContents(file_response.read().splitlines())
            
            result_file_name = "data/GPTRegens/"+human_dataset_name+'_regeneration_'+output_name+".jsonl"

            with open(result_file_name, 'a') as file:
                for f in completions:
                    file.write(json.dumps(f)+'\n')
        #Create a new file that contains the rest of the batch items which failed
        compl_ids = [x['custom_id'] for x in completions]
        not_completed = []
        with open("data/RegenerationCache/"+output_name+".jsonl", 'r') as reader:
            for f in reader:
                if len(f) > 0:
                    t = json.loads(f.strip())
                    if t['custom_id'] not in compl_ids:
                        not_completed.append(t)

        with open("data/RegenerationCache/"+output_name+".jsonl", 'w') as writer:
            for f in not_completed:
                writer.write(json.dumps(f)+'\n')
        
        
            
    else:
        result_file_id = batch_job.output_file_id
        file_response = client.files.content(result_file_id)
        completions = readBatchResponseContents(file_response.read().splitlines())
        result_file_name = "data/GPTRegens/"+human_dataset_name+'_regeneration_'+output_name+".jsonl"

        with open(result_file_name, 'a') as file:
            for f in completions:
                file.write(json.dumps(f)+'\n')
if __name__ == "__main__":
    main(sys.argv[1:])