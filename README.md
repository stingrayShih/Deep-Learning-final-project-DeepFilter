# Deep-Learning-final-project-DeepFilter



# Usage
First, download the hateful meme dataset at https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset


To install the required packages, run
```
pip install -r requirements.txt
```

For the LoRA finetuning for the DeepSeek-VL-1.3B-chat, Qwen2-VL-2B-Instruct, and OpenGVLab-InternVL2-1B models, first get into the swift folder and change the `data_path` in `swift_preprocessing.py` and `swift_preprocessing_for_inference.py`.
Then run the following command to preprocess the data into the desired prompt format for finetuning.
```
python swift_preprocessing.py
python swift_preprocessing_for_inference.py
```

For finetuning, we use the ms-swift package. To finetune a model, run
```
CUDA_VISIBLE_DEVICES=0 swift sft     --model_type <model type>  --model <model name>   --dataset <your preprocessed data path>
```

To generate latent embedding for classification, first change the `model_dir`, `adapter_dir`, and path to the preprocessed data in `get_embedding_deepseek.py`, `get_embedding_qwen2b.py`, and `get_embedding_intern.py`, then run
```
python get_embedding_deepseek.py
python get_embedding_qwen2b.py
python get_embedding_intern.py
```
Note that this step has to be run two times: one time for the training set, one time for the developement set.

To do classification, first set the `MODEL_NAME`, `TRAIN_DATA_PATH`, and `TEST_DATA_PATH` in `binary_classifier_train.py` and `random_forest_train.py`, and then run
```
python binary_classifier_train.py
python random_forest_train.py
```
To do classification with ensembling, first set the `TRAIN_DATA_PATH`, and `TEST_DATA_PATH` in `ensemble_NN.py` and `ensemble_random_forest.py`, and then run
```
ensemble_NN.py
ensemble_random_forest.py
```


