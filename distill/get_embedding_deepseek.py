from swift.llm import get_model_tokenizer, get_template, InferRequest, RequestConfig, PtEngine
from modelscope import snapshot_download
from swift.tuners import Swift


import torch
import torch.nn as nn

import json


dataset_type='dev'


# Paths
model_dir = "/path"  # <== ✅ Update this
adapter_dir = "./output/deepseek-vl-adapter/v0-20250426-195631/checkpoint-825"  # <== ✅ Update this

# Load base model and tokenizer
model, tokenizer = get_model_tokenizer(model_dir, device_map='auto')

# Load LoRA adapter
model = Swift.from_pretrained(model, adapter_dir)

# Replace lm_head to extract embeddings
model.language_model.lm_head = nn.Identity()

print(model)
print(model.language_model)

embedding_output=[]


def hook_fn(module, input, output):
    embedding_output.append(output[:, -1, :].detach().cpu().float())

hook_handle = model.language_model.lm_head.register_forward_hook(hook_fn)




with open(f'./data/{dataset_type}_swift.json', 'r') as f:
    dev_groundtruth=json.load(f)


groundtruth=[i['conversations'][1]['value'] for i in dev_groundtruth]
print(groundtruth)






template = get_template(model.model_meta.template, tokenizer)
engine = PtEngine.from_model_template(model, template)

dev_datapath = f'./data/{dataset_type}_swift_for_inference.json'

with open(dev_datapath, 'r') as f:
    dev_list = json.load(f)


request_config = RequestConfig(max_tokens=1, temperature=0)
resp_list = engine.infer([InferRequest(messages=i) for i in dev_list], request_config=request_config)
print(resp_list)

hook_handle.remove()

groundtruth_int = []
for i in groundtruth:
    if 'YES' in i:
        groundtruth_int.append(1)
    elif 'NO' in i:
        groundtruth_int.append(0)
    else:
        print("wrong value, we received", i)

print('Data Length:',len(embedding_output), len(groundtruth_int))

torch.save({'embedding':embedding_output, 'groundtruth': groundtruth_int}, f'./output/embeddings/deepseek1_3b_binary_data_{dataset_type}.pt')





