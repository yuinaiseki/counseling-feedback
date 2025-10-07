from datasets import load_dataset
    
output_dir= "/root/model_training/models/output"
dataset_name = "feedback_qesconv_labeledsplit"

print(f"Loading dataset: {dataset_name}")
try:
    dataset = load_dataset(f"./data/{dataset_name}")
    print(f"Successfully loaded dataset: {dataset_name}")
except Exception as e:
    print(f"FAILED TO LOAD DATASET: {e}")


dataset['test'] = dataset['test'].map(lambda x: {'text': f'<s>{x["text"]}',
                                                    'helper_index': x['helper_index'],
                                                    'conv_index': x['conv_index']})
