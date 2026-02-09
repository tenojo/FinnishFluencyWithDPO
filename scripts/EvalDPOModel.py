
#imports
from vllm import LLM
import json
import sys


def main(cmd_args):
    MODEL_PATH = cmd_args[0]
    llm = LLM(model=MODEL_PATH)

    eval_ds_path = cmd_args[1]
    ds_items = []
    with open(eval_ds_path, 'r', encoding="UTF-8") as reader:
        for l in reader:
            if len(l) > 1:
                ds_items.append(json.loads(l.strip()))
    ds_items = ds_items[-1000:]

    prompts = [x['prompt'] for x in ds_items]

    outputs = llm.generate(prompts)
    res_d = []
    for i,o in enumerate(outputs):
        res_d.append({'pos_id':i, 'model_pred':o.outputs[0].text, 'correct_output':ds_items[i]['chosen'], 'prompt':prompts[i]})

    print("Parsed outputs!")

    with open("data/first_eval_results.jsonl", "w") as writer:
        for d in res_d:
            writer.write(json.dumps(d)+'\n')
    
    print("done!")


if __name__ == "__main__":
    main(sys.argv[1:])