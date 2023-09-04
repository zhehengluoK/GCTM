import sys
import json
from datasets import load_dataset

dataname = sys.argv[1]
dataset = load_dataset(dataname)
train = dataset['train']

with open(f"data/{dataname.split('/')[1]}/train.jsonl", 'w') as f:
    for i, rcd in enumerate(train):
        cur_rcd = {}
        cur_rcd['id'] = rcd[str(i)]
        conv = []
        conv.append({'from': "human", "value": rcd['input']})
        conv.append({'from': "assistant", "value": rcd['output']})
        cur_rcd['conversation'] = conv
