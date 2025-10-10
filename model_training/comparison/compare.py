import json
from typing import Dict, List
from collections import defaultdict, Counter

# formatting for comparison
def normalize_feedback(feedback: Dict) -> Dict:
    normalized = feedback.copy()
    
    if "goodareas" in normalized and isinstance(normalized["goodareas"], list):
        normalized["goodareas"] = sorted(normalized["goodareas"])
    if "badareas" in normalized and isinstance(normalized["badareas"], list):
        normalized["badareas"] = sorted(normalized["badareas"])
    
    return normalized

# comparison metrics for one instance
def calculate_metrics(labeled: Dict, generated: Dict) -> Dict:
    metrics = {}

    metrics["perfect_match"] = labeled.get("perfect") == generated.get("perfect")

    labeled_good = set(labeled.get("goodareas", []))
    gen_good = set(generated.get("goodareas", []))
    
    if labeled_good or gen_good:
        intersection = len(labeled_good & gen_good)
        union = len(labeled_good | gen_good)
        
        metrics["goodareas_precision"] = intersection / len(gen_good) if gen_good else 0
        metrics["goodareas_recall"] = intersection / len(labeled_good) if labeled_good else 0
        
        if intersection > 0:
            metrics["goodareas_f1"] = (2 * intersection / (len(labeled_good) + len(gen_good)))
        else:
            metrics["goodareas_f1"] = 0
            
        metrics["goodareas_exact_match"] = labeled_good == gen_good
    else:
        metrics["goodareas_precision"] = 1.0
        metrics["goodareas_recall"] = 1.0
        metrics["goodareas_f1"] = 1.0
        metrics["goodareas_exact_match"] = True
    
    labeled_bad = set(labeled.get("badareas", []))
    gen_bad = set(generated.get("badareas", []))
    
    if labeled_bad or gen_bad:
        intersection = len(labeled_bad & gen_bad)
        union = len(labeled_bad | gen_bad)
        
        metrics["badareas_precision"] = intersection / len(gen_bad) if gen_bad else 0
        metrics["badareas_recall"] = intersection / len(labeled_bad) if labeled_bad else 0
        
        if intersection > 0:
            metrics["badareas_f1"] = (2 * intersection / (len(labeled_bad) + len(gen_bad)))
        else:
            metrics["badareas_f1"] = 0
            
        metrics["badareas_exact_match"] = labeled_bad == gen_bad

    metrics["exact_match"] = normalize_feedback(labeled) == normalize_feedback(generated)
    
    metrics["labeled_perfect"] = labeled.get("perfect")
    metrics["gen_perfect"] = generated.get("perfect")
    metrics["labeled_goodareas"] = sorted(labeled_good) if labeled_good else []
    metrics["gen_goodareas"] = sorted(gen_good) if gen_good else []
    
    if labeled_bad or gen_bad:
        metrics["labeled_badareas"] = sorted(labeled_bad) if labeled_bad else []
        metrics["gen_badareas"] = sorted(gen_bad) if gen_bad else []
    
    return metrics

# checks if model struggles with any particular category
def per_category_performance(data: List[Dict]) -> Dict:
    """Calculate performance metrics for each category."""
    # Collect all categories
    all_categories = set()
    for d in data:
        all_categories.update(d['labeled'].get('goodareas', []))
        all_categories.update(d['labeled'].get('badareas', []))
    
    category_metrics = {}
    
    for category in all_categories:
        tp = sum(1 for d in data 
                if category in d['labeled'].get('goodareas', []) and 
                   category in d['generated'].get('goodareas', []))
        
        fp = sum(1 for d in data 
                if category not in d['labeled'].get('goodareas', []) and 
                   category in d['generated'].get('goodareas', []))
        
        fn = sum(1 for d in data 
                if category in d['labeled'].get('goodareas', []) and 
                   category not in d['generated'].get('goodareas', []))
        
        tn = len(data) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(data) if len(data) > 0 else 0
        
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'support': tp + fn,  # Number of times this category appears in ground truth
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    return category_metrics

# compares TP, TN, FP, FN in comparison
def confusion_matrix_analysis(data: List[Dict]) -> Dict:
    tp = sum(1 for d in data if d['labeled'].get('perfect') == True and d['generated'].get('perfect') == True)
    tn = sum(1 for d in data if d['labeled'].get('perfect') == False and d['generated'].get('perfect') == False)
    fp = sum(1 for d in data if d['labeled'].get('perfect') == False and d['generated'].get('perfect') == True)
    fn = sum(1 for d in data if d['labeled'].get('perfect') == True and d['generated'].get('perfect') == False)
    
    total = tp + tn + fp + fn
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp + tn) / total if total > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }

def difficulty_analysis(data: List[Dict]) -> Dict:
    by_num_categories = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    
    for d in data:
        num_categories = len(d['labeled'].get('goodareas', []))
        is_correct = set(d['labeled'].get('goodareas', [])) == set(d['generated'].get('goodareas', []))
        
        by_num_categories[num_categories]['total'] += 1
        if is_correct:
            by_num_categories[num_categories]['correct'] += 1
        
        by_num_categories[num_categories]['examples'].append({
            'helper_index': d['helper_index'],
            'conv_index': d['conv_index'],
            'correct': is_correct
        })
    
    # Calculate accuracy for each difficulty level
    difficulty = {}
    for num_categories, stats in by_num_categories.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        difficulty[num_categories] = {
            'accuracy': accuracy,
            'correct': stats['correct'],
            'total': stats['total'],
            'examples_sampled': stats['examples'][:5]  # Keep only first 5 examples
        }
    
    return difficulty

def error_analysis(data: List[Dict]) -> Dict:
    """Analyze common error patterns."""
    errors = {
        'fp': [],
        'fn': [],
        'hallucinated_categories': Counter(),
        'omitted_categories': Counter(),
    }
    
    for d in data:
        gt = d['labeled']
        gen = d['generated']
        
        # False positives/negatives for perfect field
        if gt.get('perfect') != gen.get('perfect'):
            if gen.get('perfect') == True:
                errors['fp'].append({
                    'helper_index': d['helper_index'],
                    'conv_index': d['conv_index']
                })
            else:
                errors['fn'].append({
                    'helper_index': d['helper_index'],
                    'conv_index': d['conv_index']
                })
        
        # Category errors
        gt_good = set(gt.get('goodareas', []))
        gen_good = set(gen.get('goodareas', []))
        
        hallucinated = gen_good - gt_good
        omitted = gt_good - gen_good
        
        if hallucinated:
            errors['hallucinated_categories'].update(hallucinated)
        
        if omitted:
            errors['omitted_categories'].update(omitted)
    
    return {
        'fp_count': len(errors['fp']),
        'fn_count': len(errors['fn']),
        'fp_examples': errors['fp'][:5],
        'fn_examples': errors['fn'][:5],
        'most_hallucinated': dict(errors['hallucinated_categories'].most_common(10)),
        'most_omitted': dict(errors['omitted_categories'].most_common(10))
    }

def compare_results(combined_file: str, output_file: str = "comparison_results.json"):
    """Compare ground truth and generated results."""
    
    with open(combined_file, 'r') as f:
        data = json.load(f)
    
    print(f"Comparing {len(data)} examples...\n")
   
    all_metrics = []
    for i in data:
        metrics = calculate_metrics(i["labeled"], i["generated"])
        
        metrics["helper_index"] = i["helper_index"]
        metrics["conv_index"] = i["conv_index"]
        
        all_metrics.append(metrics)
    
    n = len(all_metrics)
    
    aggregated = {
        "total_examples": n,
        "exact_match_accuracy": sum(m["exact_match"] for m in all_metrics) / n * 100,
        "perfect_field_accuracy": sum(m["perfect_match"] for m in all_metrics) / n * 100,
        "goodareas_exact_match": sum(m["goodareas_exact_match"] for m in all_metrics) / n * 100,
        "goodareas_precision": sum(m["goodareas_precision"] for m in all_metrics) / n * 100,
        "goodareas_recall": sum(m["goodareas_recall"] for m in all_metrics) / n * 100,
        "goodareas_f1": sum(m["goodareas_f1"] for m in all_metrics) / n * 100,
    }
    
    has_badareas = any("badareas_f1" in m for m in all_metrics)
    if has_badareas:
        badareas_metrics = [m for m in all_metrics if "badareas_f1" in m]
        nb = len(badareas_metrics)
        aggregated["badareas_exact_match"] = sum(m["badareas_exact_match"] for m in badareas_metrics) / nb * 100
        aggregated["badareas_precision"] = sum(m["badareas_precision"] for m in badareas_metrics) / nb * 100
        aggregated["badareas_recall"] = sum(m["badareas_recall"] for m in badareas_metrics) / nb * 100
        aggregated["badareas_f1"] = sum(m["badareas_f1"] for m in badareas_metrics) / nb * 100

    confusion_matrix = confusion_matrix_analysis(data)
    category_performance = per_category_performance(data)
    difficulty = difficulty_analysis(data)
    error = error_analysis(data)
    
    print("EVALUATION RESULTS")
    print(f"\n1. Overall Metrics:")
    print(f" - Total Examples: {aggregated['total_examples']}")
    print(f" - Exact Match Accuracy: {aggregated['exact_match_accuracy']:.2f}%")
    print(f" - Perfect Field Accuracy: {aggregated['perfect_field_accuracy']:.2f}%")

    mismatches = [m for m in all_metrics if not m["exact_match"]]
    print(f" - Non-exact matches: {len(mismatches)} / {n} ({len(mismatches)/n*100:.1f}%)")
  
    print(f"\n2. Good Areas:")
    print(f" - Exact: {aggregated['goodareas_exact_match']:.2f}%")
    print(f" - Precision: {aggregated['goodareas_precision']:.2f}%")
    print(f" - Recall: {aggregated['goodareas_recall']:.2f}%")
    print(f" - F1 Score: {aggregated['goodareas_f1']:.2f}%")
    
    if has_badareas:
        print(f"\n3. Bad Areas:")
        print(f" - Ecxact: {aggregated['badareas_exact_match']:.2f}%")
        print(f" - Precision: {aggregated['badareas_precision']:.2f}%")
        print(f" - Recall: {aggregated['badareas_recall']:.2f}%")
        print(f" - F1 Score: {aggregated['badareas_f1']:.2f}%")
    
    print(f"\n3. Confusion Matrix:")
    print(f" - TP:  {confusion_matrix['tp']}")
    print(f" - TM:  {confusion_matrix['tn']}")
    print(f" - FP: {confusion_matrix['fp']}")
    print(f" - FN: {confusion_matrix['fn']}")
    print(f" - Accuracy (weight of TP and TN): {confusion_matrix['accuracy']:.2%}")
    print(f" - Precision (when perfect = true, how many is correct): {confusion_matrix['precision']:.2%}")
    print(f" - Recall (of all perfect = true cases, how many was correct): {confusion_matrix['recall']:.2%}")

    
    print(f"\n4. Difficulty Analysis (by number of categories):")
    for num_categories in sorted(difficulty.keys()):
        stats = difficulty[num_categories]
        print(f" - {num_categories} categories: {stats['accuracy']:.2%} accuracy "
              f"({stats['correct']}/{stats['total']} correct)")
    
    print(f"\n5. Error Analysis:")
    print(f" - False Positives (said perfect = True, wasn't): {error['fp_count']}")
    print(f" - False Negatives (said perfect = False, was perfect): {error['fn_count']}")
    
    if error['most_hallucinated']:
        print(f"\n  Most Hallucinated:")
        for category, count in list(error['most_hallucinated'].items())[:5]:
            print(f" -   {category}: {count} times")
    
    if error['most_omitted']:
        print(f"\n  Most Omitted:")
        for category, count in list(error['most_omitted'].items())[:5]:
            print(f" -   {category}: {count} times") 

    print(f"\n6. Per-Category Performance:")

    sorted_categories = sorted(category_performance.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for category, metrics in sorted_categories:
        print(f" - {category:20s} - TP / (TP + FP): {metrics['precision']:.2%}, TP / (TP + FN): {metrics['recall']:.2%}, "
              f"TP + TN / All: {metrics['accuracy']:.2%}")

    # save
    results = {
        "summary": aggregated,
        "per_example": all_metrics,
        "mismatches": mismatches,
        "confusion_matrix": confusion_matrix,
        "per_category": category_performance,
        "difficulty": difficulty,
        "errors": error
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    compare_results("comparison.json")