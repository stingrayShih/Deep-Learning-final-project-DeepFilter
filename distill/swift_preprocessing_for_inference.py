import json
from qqdm import qqdm


def preprocess(data_path, json_file):
    data=[]
    
    with open(data_path+json_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    processed=[]
    

    for ex in qqdm(data):
        img_path = data_path + ex['img']
        text     = ex['text']

        processed.append([{
            "role": "user",
            "content":
            "Is this meme hateful? Answer ONLY YES or NO.\n"
            f"Meme image:<img>{img_path}</img>\nMeme text: {text}"
        }])

    # dev_swift_for_inference.json
    with open(data_path+json_file[:-6]+'_swift_for_inference.json', 'w') as f:
        json.dump(processed, f, indent=4)
    print(f'saved as {data_path+json_file[:-6]}'+'_swift_for_inference.json')



if __name__=="__main__":
    print("processing dataset for inference")

    data_path='data/'
    json_files=['train.jsonl', 'dev.jsonl']

    for json_file in json_files:
        preprocess(data_path, json_file)



    print("process complete!")
