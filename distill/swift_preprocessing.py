import json
from qqdm import qqdm


def preprocess(data_path, jsonl_file):
    data=[]
    with open(data_path+jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    processed=[]
    
    for ex in qqdm(data):
        img_path = data_path + ex['img']
        text     = ex['text']
        label    = 'YES' if ex['label'] else 'NO'
        reason   = ex.get('gemini_reason', '')      # ← NEW

        user_msg = (
            "Determine whether this meme is hateful or not. "
            "First give a 1-2 sentence explanation, then on a new line write "
            "FINAL ANSWER: YES or NO.\n"
            f"Meme image:<img>{img_path}</img>\nMeme text: {text}"
        )
        #assistant_msg = reason + f"\n\nFINAL ANSWER: {label}"
        assistant_msg = (
            reason +
            "\n\n<ans_token> "          #  ← sentinel marks where we embed
            f"{label}"
        )
        processed.append({
            "conversations": [
                {"from": "user",       "value": user_msg},
                {"from": "assistant",  "value": assistant_msg}
            ]
        })

    with open(data_path+jsonl_file[:-6]+'_swift.json', 'w') as f:
        json.dump(processed, f, indent=4)
    print(f'saved as {data_path+jsonl_file[:-6]}'+'_swift.json')


"""
training command:
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model_type deepseek_vl \
  --model ./modelpath\
  --dataset ./data/train_swift.json \
  --output_dir ./output/deepseek-vl-adapter \
  --lora_r 8 --lora_alpha 32 --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --fp16 true \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 2


"""


if __name__=="__main__":
    print("preprocessing dataset")

    data_path='data/'  # /Desktop/Deep learning/Deepseek/
    jsonl_files=['train.jsonl', 'dev.jsonl']


    for jsonl_file in jsonl_files:
        preprocess(data_path, jsonl_file)



    print("preprocess complete!")
