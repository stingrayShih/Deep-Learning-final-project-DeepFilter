import json
from qqdm import qqdm


def preprocess(data_path, jsonl_file):
    data=[]
    with open(data_path+jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    processed=[]
    
    for example in qqdm(data):
        image_path=data_path+example['img']
        text=example['text']
        if example['label']==0:
            label='NO'
        elif example['label']==1:
            label='YES'
        else:
            raise

        conversation={"conversations": [
        {"from": "user", "value": f"Determine the following meme is hateful or not. Please answer with only YES or NO. \n Meme image:<img>{image_path}</img>\n Meme text: {text}"},
        {"from": "assistant", "value": label}
        ]}

        processed.append(conversation)
    with open(data_path+jsonl_file[:-6]+'_swift.json', 'w') as f:
        json.dump(processed, f, indent=4)
    print(f'saved as {data_path+jsonl_file[:-6]}'+'_swift.json')


'''
training command:
CUDA_VISIBLE_DEVICES=0 swift sft     --model_type deepseek_vl  --model deepseek-ai/deepseek-vl-1.3b-chat   --dataset /home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/train_swift.json --resume_from_checkpoint output/deepseek-vl-1.3b-chat/v1-20250308-144113/checkpoint-24500

CUDA_VISIBLE_DEVICES=0 swift sft     --model_type qwen2_vl  --model qwen/Qwen2-VL-2B-Instruct   --dataset /home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/train_swift.json --resume_from_checkpoint output/Qwen2-VL-2B-Instruct/v2-20250411-205338/checkpoint-18000

CUDA_VISIBLE_DEVICES=0 swift sft     --model_type internvl2  --model OpenGVLab/InternVL2-1B   --dataset /home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/train_swift.json --resume_from_checkpoint output/InternVL2-1B/v2-20250417-131126/checkpoint-22000
'''




if __name__=="__main__":
    print("preprocessing dataset")

    data_path='/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/'
    jsonl_files=['train.jsonl', 'dev.jsonl']


    for jsonl_file in jsonl_files:
        preprocess(data_path, jsonl_file)



    print("preprocess complete!")
