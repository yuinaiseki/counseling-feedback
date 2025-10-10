import json

def extract_prompt(text: str) -> str:
    end_idx = text.find("### Response:")
    if end_idx == -1:
        return text
    return text[:end_idx].strip()

def extract_labeled_json(text: str) -> dict:
    start = text.find("### Response:") + len("### Response:")
    json_str = text[start:].strip()
    return json.loads(json_str)

def combine_data(test_labeled: str, generated: str, output: str):
    
    # Load data
    with open(test_labeled, 'r') as f:
        labeled = json.load(f)
    
    with open(generated, 'r') as f:
        predictions = json.load(f)
    
    # Create lookup dict for predictions by (helper_index, conv_index)
    pred_lookup = {}
    for pred in predictions:
        key = (pred['helper_index'], pred['conv_index'])
        pred_lookup[key] = pred
    
    # Combine
    combined = []
    for i in labeled:
        key = (i['helper_index'], i['conv_index'])
        
        if key not in pred_lookup:
            print(f"Warning: No prediction found for helper_index={i['helper_index']}, conv_index={i['conv_index']}")
            continue
        
        pred = pred_lookup[key]
        
        # Extract components
        prompt = extract_prompt(i['text'])
        labeled_response = extract_labeled_json(i['text'])
        
        # Get generated response
        if isinstance(pred.get('output'), dict):
            generated_response = pred['output']
        else:
            generated_response = json.loads(pred['raw_output'])
        
        # Create obj
        combined_obj = {
            "helper_index": i['helper_index'],
            "conv_index": i['conv_index'],
            "prompt": prompt,
            "labeled": labeled_response,
            "generated": generated_response
        }
        
        combined.append(combined_obj)
    
    # Save to file
    with open(output, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"Combined {len(combined)} examples")
    print(f"Saved to {output}")
    
    # Print eg
    if combined:
        print("\nFirst example:")
        print(json.dumps(combined[0], indent=2))

if __name__ == "__main__":
    combine_data(
        test_labeled="test_og.json",
        generated="generations.json", 
        output="comparison.json"
    )