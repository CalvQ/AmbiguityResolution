#!/usr/bin/env python3
"""
Standalone evaluation script for AREngine ambiguity detection.
Does not require scene_parser - uses raw environment_info.

Usage:
    python eval_dataset_simple.py <dataset.json> [output.csv] [limit]
    
Example:
    python eval_dataset_simple.py data.json results.csv 100
"""

import json
import csv
from pathlib import Path
from tqdm.auto import tqdm

from arengine import AREngine


def evaluate_json_file(json_path, out_csv="results.csv", limit=None):
    """
    Evaluate AREngine ambiguity detection on a dataset.
    
    Args:
        json_path (str): Path to JSON dataset
        out_csv (str): Output CSV path for detailed results
        limit (int): Optional limit on number of samples to evaluate
    """
    data = json.loads(Path(json_path).read_text())
    
    # Respect an optional limit
    items = data if limit is None else data[:limit]
    total = len(items)
    
    print(f"Loading model...")
    engine = AREngine()
    engine.load_model()
    print(f"Model loaded! Evaluating {total} samples...\n")

    results = []
    
    # Counters for overall stats
    n_total = 0
    n_correct = 0
    
    # Counters by ground truth type
    n_positive = 0  # should be ambiguous
    n_negative = 0  # should NOT be ambiguous
    
    # Counters for predictions
    n_pred_true = 0   # predicted ambiguous
    n_pred_false = 0  # predicted not ambiguous
    n_pred_none = 0   # model output unparseable
    
    # Counters for correct predictions by type
    n_positive_correct = 0  # correctly identified as ambiguous
    n_negative_correct = 0  # correctly identified as not ambiguous

    for item in tqdm(items, desc="Evaluating", unit="scene"):
        scene_id = item.get("scene_id", "unknown")
        object_id = item.get("object_id", "unknown")
        object_name = item.get("object_name", "unknown")
        
        # Get the prompt (first dialogue turn)
        dialogue = item.get("dialogue", [])
        if not dialogue:
            print(f"\nWarning: No dialogue found for scene {scene_id}")
            continue
        prompt = dialogue[0].get("text", "")
        
        # Get ground truth from sample_type
        sample_type = item.get("sample_type", "").lower()
        if sample_type == "positive":
            gt = True  # Should be detected as ambiguous
            n_positive += 1
        elif sample_type == "negative":
            gt = False  # Should NOT be detected as ambiguous
            n_negative += 1
        else:
            print(f"\nWarning: Unknown sample_type '{sample_type}' for scene {scene_id}")
            continue
        
        # Get scene (use raw environment info)
        env_info = (item.get("environment_info") or "").strip()
        scene = {"environment_info": env_info}
        
        # Run ambiguity detection
        pred, raw = engine.is_prompt_ambiguous(prompt, scene=scene)
        
        # Update counters
        n_total += 1
        if pred is True:
            n_pred_true += 1
        elif pred is False:
            n_pred_false += 1
        else:
            n_pred_none += 1
        
        # Check if prediction is correct
        is_correct = (pred == gt)
        if is_correct:
            n_correct += 1
            if gt:
                n_positive_correct += 1
            else:
                n_negative_correct += 1
        
        # Store result
        results.append({
            "scene_id": scene_id,
            "object_id": object_id,
            "object_name": object_name,
            "sample_type": sample_type,
            "ambiguity_type": item.get("ambiguity_type", ""),
            "initial_prompt": prompt,
            "predicted_ambiguous": pred,
            "ground_truth_ambiguous": gt,
            "correct": is_correct,
            "raw_model_output": raw,
        })

    # Write detailed CSV results
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "object_id",
                "object_name",
                "sample_type",
                "ambiguity_type",
                "initial_prompt",
                "predicted_ambiguous",
                "ground_truth_ambiguous",
                "correct",
                "raw_model_output",
            ],
        )
        w.writeheader()
        w.writerows(results)

    # Calculate metrics
    overall_acc = (n_correct / n_total * 100) if n_total > 0 else 0.0
    positive_acc = (n_positive_correct / n_positive * 100) if n_positive > 0 else 0.0
    negative_acc = (n_negative_correct / n_negative * 100) if n_negative > 0 else 0.0
    
    # Calculate precision, recall, F1 for ambiguous class
    # True Positive: predicted True, gt True
    # False Positive: predicted True, gt False
    # False Negative: predicted False, gt True
    tp = n_positive_correct
    fp = n_pred_true - tp  # predicted true but was actually false
    fn = n_positive - tp   # should be true but predicted false
    
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal samples evaluated: {n_total}")
    print(f"  - Positive samples (should be ambiguous): {n_positive}")
    print(f"  - Negative samples (should NOT be ambiguous): {n_negative}")
    
    print(f"\nPredictions:")
    print(f"  - Predicted TRUE (ambiguous): {n_pred_true}")
    print(f"  - Predicted FALSE (not ambiguous): {n_pred_false}")
    print(f"  - Predicted NONE (parse error): {n_pred_none}")
    
    print(f"\nAccuracy:")
    print(f"  - Overall: {overall_acc:.1f}% ({n_correct}/{n_total})")
    print(f"  - Positive samples: {positive_acc:.1f}% ({n_positive_correct}/{n_positive})")
    print(f"  - Negative samples: {negative_acc:.1f}% ({n_negative_correct}/{n_negative})")
    
    print(f"\nMetrics for Ambiguous Detection:")
    print(f"  - Precision: {precision:.1f}%")
    print(f"  - Recall: {recall:.1f}%")
    print(f"  - F1 Score: {f1:.1f}%")
    
    print(f"\nDetailed results saved to: {out_csv}")
    print("=" * 70)
    
    return {
        "total": n_total,
        "correct": n_correct,
        "overall_accuracy": overall_acc,
        "positive_accuracy": positive_acc,
        "negative_accuracy": negative_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval_dataset_simple.py <dataset.json> [output.csv] [limit]")
        print("\nExample:")
        print("  python eval_dataset_simple.py data.json results.csv 100")
        sys.exit(1)
    
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results.csv"
    
    # Optional limit
    limit = None
    if len(sys.argv) > 3:
        try:
            limit = int(sys.argv[3])
        except ValueError:
            print(f"Warning: Could not parse limit '{sys.argv[3]}', evaluating all samples")
    
    evaluate_json_file(in_path, out_csv=out_path, limit=limit)