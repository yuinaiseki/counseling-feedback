"""
Running generation on pre-existing avylor-feedback model.

To run this script, ensure you have Modal installed and configured. 
Make sure you are in the model_training directory. Then run
```bash
modal run generate_modal.py --dataset-name "your_dataset" --threshold 0.6
```
"""

import modal
import json
from typing import Optional

# Create a Modal stub
app = modal.App("avylor-feedback")

# Define the image with all dependencies and pre-downloaded models
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "accelerate",
    )
    .add_local_dir(".", remote_path="/root/model_training", ignore=["venv", "__pycache__", "*.git*"])
)

# Volume for caching models (optional but recommended for faster cold starts)
MODEL_VOLUME = modal.Volume.from_name("avylor-feedback", create_if_missing=True)

MAX_TRIES = 5


def ann_check(ann):
    if "goodareas" not in ann:
        raise Exception("No goodareas in annotation!")
    if "perfect" not in ann:
        raise Exception("No perfect in annotation!")
    if ann["perfect"] == False:
        if "feedback" not in ann:
            raise Exception("No feedback in annotation!")
        if "badareas" not in ann:
            raise Exception("No areas in annotation!")
        if "alternative" not in ann:
            raise Exception("No alternative in annotation!")


def extract_output(s):
    start_index = s.find("Response:")
    start_index += len("Response:")
    extracted_string = s[start_index:]
    return extracted_string


@app.function(
    image=image,
    gpu="A10G",  # Options: "T4", "A10G", "A100", "H100"
    timeout=7200,  # 2 hours
    volumes={"/cache": MODEL_VOLUME},
)
def generate_feedback_batch(
    start_index: int = 0,
    end_index: Optional[int] = None,
    dataset_name: str = "feedback_qesconv_labeledsplit",
    threshold: float = 0.5,
):
    """Generate feedback for a batch of examples."""
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from datasets import load_dataset
    from transformers import set_seed
    from tqdm import tqdm
    
    set_seed(42)
    torch.manual_seed(42)
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [29913, 12258, 500, 2]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "avylor/mh_feedback_model",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir="/cache"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "avylor/mh_feedback_model",
        cache_dir="/cache"
    )
    
    print("Loading dataset...")
    try:
        dataset = load_dataset(
            "json",
            data_files={
                "train": f"/root/model_training/data/{dataset_name}/train.json",
                "test": f"/root/model_training/data/{dataset_name}/test.json"
            },
            split={"train": "train", "test": "test"},
            cache_dir="/root/model_training/cache"
        )
        print(f"Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"FAILED TO LOAD DATASET: {e}")
        return None
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    dataset['test'] = dataset['test'].map(
        lambda x: {
            'text': f'<s>{x["text"]}',
            'helper_index': x['helper_index'],
            'conv_index': x['conv_index']
        }
    )
    
    model.eval()
    
    if end_index is None:
        end_index = len(dataset['test'])
    
    generations = []
    
    with torch.no_grad():
        for ind in tqdm(range(start_index, end_index)):
            print(f"\n--------------------- Example {ind} -----------------------")
            print("Input:")
            print(dataset['test'][ind]['text'])
            print("--------------------- Output -----------------------")
            
            helper_line = dataset['test'][ind]['text'].split('Response:')[0].split('\n')[-3]
            original_feedback = dataset['test'][ind]['text'] + json.dumps({"perfect": True})[:-6]
            new_prompt_encoded = tokenizer(
                original_feedback,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model(**new_prompt_encoded)
            logits = outputs.logits
            
            last_token_logits = logits[0, -1, :]
            probabilities = softmax(last_token_logits, dim=0)
            max_prob_index = torch.argmax(probabilities).item()
            predicted_token = tokenizer.convert_ids_to_tokens(max_prob_index)
            t_index = tokenizer.convert_tokens_to_ids('▁true')
            probability_of_t = probabilities[t_index].item()
            
            if probability_of_t > threshold:
                feedback_to_continue = original_feedback + ' true'
            else:
                feedback_to_continue = original_feedback + ' false'
            
            query = tokenizer(
                feedback_to_continue,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            
            output = model.generate(
                **query,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()])
            )
            
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            feedback_only = extract_output(decoded_output)
            
            attempt = 0
            parsed_output = None
            while attempt < MAX_TRIES:
                try:
                    loaded_output = json.loads(feedback_only)
                    ann_check(loaded_output)
                    print(loaded_output)
                    parsed_output = loaded_output
                    break
                except Exception as e:
                    print(e)
                    print(f'### Attempt {attempt} Failed to parse output as json\n\n')
                    attempt += 1
            
            generations.append({
                'index': ind,
                'helper_index': dataset['test'][ind]['helper_index'],
                'conv_index': dataset['test'][ind]['conv_index'],
                'input': dataset['test'][ind]['text'],
                'output': parsed_output,
                'raw_output': feedback_only,
                'probability_of_t': probability_of_t
            })
    
    return generations


@app.function(image=image)
def parallel_generate(
    total_examples: int,
    num_workers: int = 4,
    dataset_name: str = "feedback_qesconv_labeledsplit",
    threshold: float = 0.5,
):
    """Split work across multiple GPUs in parallel."""
    batch_size = (total_examples + num_workers - 1) // num_workers
    
    # Create batch ranges
    ranges = []
    for i in range(num_workers):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_examples)
        if start < total_examples:
            ranges.append((start, end))
    
    print(f"Splitting {total_examples} examples across {len(ranges)} workers")
    for i, (start, end) in enumerate(ranges):
        print(f"Worker {i}: examples {start}-{end}")
    
    # Run in parallel using Modal's .map()
    results = []
    for start, end in ranges:
        result = generate_feedback_batch.remote(
            start_index=start,
            end_index=end,
            dataset_name=dataset_name,
            threshold=threshold
        )
        results.append(result)
    
    # Collect all results
    all_generations = []
    for result in results:
        all_generations.extend(result)
    
    return all_generations


@app.local_entrypoint()
def main(
    start_index: int = 0,
    end_index: Optional[int] = None,
    dataset_name: str = "feedback_qesconv_labeledsplit",             #dataset
    threshold: float = 0.5,
    parallel: bool = False,
    num_workers: int = 4,
):

    if parallel:
        if end_index is None:
            raise ValueError("Must specify --end-index when using --parallel")
        results = parallel_generate.remote(
            total_examples=end_index,
            num_workers=num_workers,
            dataset_name=dataset_name,
            threshold=threshold
        )
    else:
        results = generate_feedback_batch.remote(
            start_index=start_index,
            end_index=end_index,
            dataset_name=dataset_name,
            threshold=threshold
        )

    output_file = f"generations_{start_index}_{end_index or 'end'}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Generated {len(results)} examples")
    print(f"Saved to {output_file}")