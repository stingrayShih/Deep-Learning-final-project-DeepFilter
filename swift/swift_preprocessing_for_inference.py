import json
from qqdm import qqdm


def preprocess(data_path, json_file):
    data=[]
    with open(data_path+json_file, "r") as f:
        data = json.load(f)

    processed=[]
    
    for example in qqdm(data):
        conversation=example['conversations']
        new=[{}]
        
        new[0]['role']=conversation[0]['from']
        new[0]['content']=conversation[0]['value']


        processed.append(new)


    with open(data_path+json_file[:-5]+'_for_inference.json', 'w') as f:
        json.dump(processed, f, indent=4)
    print(f'saved as {data_path+json_file[:-6]}'+'_for_inference.json')





if __name__=="__main__":
    print("processing dataset for inference")

    data_path='/home/hice1/hshih35/.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/'
    json_files=['train_swift.json', 'dev_swift.json']


    for json_file in json_files:
        preprocess(data_path, json_file)



    print("process complete!")
