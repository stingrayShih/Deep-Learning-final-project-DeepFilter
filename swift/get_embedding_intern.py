from swift.llm import get_model_tokenizer, get_template, InferRequest, RequestConfig, PtEngine
from modelscope import snapshot_download
from swift.tuners import Swift


import torch
import torch.nn as nn

import json


dataset_type='dev'


model_dir = "/storage/ice1/8/3/hshih35/.cache/modelscope/hub/models/OpenGVLab/InternVL2-1B"
adapter_dir = 'output/InternVL2-1B/v3-20250417-192831/checkpoint-25000'

model, tokenizer = get_model_tokenizer(model_dir, device_map='auto')
model = Swift.from_pretrained(model, adapter_dir)

model.model.language_model.lm_head = nn.Identity()

#print(model)
#print(model.model.lm_head)



embedding_output=[]
def hook_fn(module, input, output):
    embedding_output.append(output[:, -1, :].detach().cpu().float())


hook_handle = model.model.language_model.lm_head.register_forward_hook(hook_fn)

print(model.model.language_model)








with open(f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{dataset_type}_swift.json', 'r') as f:
    dev_groundtruth=json.load(f)


groundtruth=[i['conversations'][1]['value'] for i in dev_groundtruth]
print(groundtruth)






template = get_template(model.model_meta.template, tokenizer)
engine = PtEngine.from_model_template(model, template)

'''
messages = [{
    'role': 'system',
    'content': 'You are a helpful assistant.'
}, {
    'role': 'user',
    'content': 'who are you?'
}]
'''
dev_datapath=f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/{dataset_type}_swift_for_inference.json'
with open(dev_datapath, 'r') as f:
    dev_list = json.load(f)


#####################
#dev_list=dev_list[:20]
#groundtruth=groundtruth[:20]
#####################



request_config = RequestConfig(max_tokens=1, temperature=0)
resp_list = engine.infer([InferRequest(messages=i) for i in dev_list], request_config=request_config)
print(resp_list)




hook_handle.remove()

groundtruth_int = []
for i in groundtruth:
    if i =='YES':
        groundtruth_int.append(1)
    elif i == 'NO':
        groundtruth_int.append(0)
    else:
        raise ValueError(f'Wrong Value: \"{i}\"')

print('Data Length:',len(embedding_output), len(groundtruth_int))


torch.save({'embedding':embedding_output, 'groundtruth': groundtruth_int}, f'/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/intern1b_binary_data_{dataset_type}.pt')





