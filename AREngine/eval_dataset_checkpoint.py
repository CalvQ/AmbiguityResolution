#!/usr/bin/env python3
"""
Checkpointed evaluation script for AREngine ambiguity detection.
Saves progress periodically and can resume from checkpoint.

Usage:
    python eval_dataset_checkpoint.py <dataset.json> <output.csv> [--limit N] [--checkpoint-every N] [--resume]
    
Example:
    # Start new evaluation
    python eval_dataset_checkpoint.py data.json results.csv --checkpoint-every 10
    
    # Resume from checkpoint
    python eval_dataset_checkpoint.py data.json results.csv --resume
"""

import json
import csv
import pickle
from pathlib import Path
from tqdm.auto import tqdm
import sys
import argparse
from datetime import datetime

from arengine import AREngine


def save_checkpoint(checkpoint_path, state):
    """Save checkpoint to disk"""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"\n[Checkpoint saved: {checkpoint_path}]")


def load_checkpoint(checkpoint_path):
    """Load checkpoint from disk"""
    if not Path(checkpoint_path).exists():
        return None
    
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    print(f"[Resuming from checkpoint: {checkpoint_path}]")
    print(f"[Progress: {state['processed_count']}/{state['total_count']} samples]")
    return state


def evaluate_with_checkpoints(
    json_path, 
    out_csv="results.csv", 
    limit=None, 
    checkpoint_every=50,
    resume=False,
    log_file=None
):
    """
    Evaluate with periodic checkpointing.
    
    Args:
        json_path (str): Path to JSON dataset
        out_csv (str): Output CSV path
        limit (int): Optional limit on samples
        checkpoint_every (int): Save checkpoint every N samples
        resume (bool): Resume from checkpoint if exists
        log_file (str): Optional log file path
    """
    # Setup logging
    if log_file:
        log_fp = open(log_file, 'a')
        def log(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{timestamp}] {msg}"
            print(log_msg)
            log_fp.write(log_msg + "\n")
            log_fp.flush()
    else:
        def log(msg):
            print(msg)
    
    # Checkpoint file path
    checkpoint_path = out_csv.replace('.csv', '_checkpoint.pkl')
    
    # Load data
    log(f"Loading dataset from {json_path}...")
    data = json.loads(Path(json_path).read_text())
    items = data if limit is None else data[:limit]
    total = len(items)
    log(f"Dataset loaded: {total} samples")
    
    # Try to resume from checkpoint
    state = None
    if resume:
        state = load_checkpoint(checkpoint_path)
    
    # Initialize or restore state
    if state is None:
        log("Starting new evaluation...")
        log("Loading model...")
        engine = AREngine()
        engine.load_model()
        log("Model loaded!")
        
        state = {
            'processed_count': 0,
            'total_count': total,
            'results': [],
            'n_total': 0,
            'n_correct': 0,
            'n_positive': 0,
            'n_negative': 0,
            'n_pred_true': 0,
            'n_pred_false': 0,
            'n_pred_none': 0,
            'n_positive_correct': 0,
            'n_negative_correct': 0,
        }
    else:
        log("Resuming from checkpoint...")
        log("Loading model...")
        engine = AREngine()
        engine.load_model()
        log("Model loaded!")
    
    start_idx = state['processed_count']
    
    # Process items
    log(f"\nProcessing samples {start_idx} to {total}...")
    
    for i in tqdm(
        range(start_idx, total),
        initial=start_idx,
        total=total,
        desc="Evaluating",
        unit="scene"
    ):
        item = items[i]
        
        scene_id = item.get("scene_id", "unknown")
        object_id = item.get("object_id", "unknown")
        object_name = item.get("object_name", "unknown")
        
        # Get the prompt
        dialogue = item.get("dialogue", [])
        if not dialogue:
            log(f"Warning: No dialogue for scene {scene_id}, skipping")
            continue
        prompt = dialogue[0].get("text", "")
        
        # Get ground truth from ambiguity_type
        ambiguity_type = item.get("ambiguity_type", "").lower()
        if ambiguity_type == "no_ambiguity":
            gt = False
            state['n_negative'] += 1
        elif ambiguity_type and ambiguity_type != "":
            gt = True
            state['n_positive'] += 1
        else:
            log(f"Warning: Missing ambiguity_type for scene {scene_id}, skipping")
            continue
        
        sample_type = item.get("sample_type", "")
        
        # Get scene
        env_info = (item.get("environment_info") or "").strip()
        scene = {"environment_info": env_info}
        
        # Run ambiguity detection
        try:
            pred, raw = engine.is_prompt_ambiguous(prompt, scene=scene)
        except Exception as e:
            log(f"Error processing scene {scene_id}: {e}")
            pred, raw = None, str(e)
        
        # Update counters
        state['n_total'] += 1
        if pred is True:
            state['n_pred_true'] += 1
        elif pred is False:
            state['n_pred_false'] += 1
        else:
            state['n_pred_none'] += 1
        
        # Check correctness
        is_correct = (pred == gt)
        if is_correct:
            state['n_correct'] += 1
            if gt:
                state['n_positive_correct'] += 1
            else:
                state['n_negative_correct'] += 1
        
        # Store result
        state['results'].append({
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
        
        state['processed_count'] = i + 1
        
        # Checkpoint periodically
        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, state)
            log(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
    
    # Final checkpoint
    save_checkpoint(checkpoint_path, state)
    
    # Write final CSV
    log(f"\nWriting results to {out_csv}...")
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
        w.writerows(state['results'])
    
    # Calculate metrics
    n_total = state['n_total']
    n_correct = state['n_correct']
    n_positive = state['n_positive']
    n_negative = state['n_negative']
    n_positive_correct = state['n_positive_correct']
    n_negative_correct = state['n_negative_correct']
    n_pred_true = state['n_pred_true']
    n_pred_false = state['n_pred_false']
    n_pred_none = state['n_pred_none']
    
    overall_acc = (n_correct / n_total * 100) if n_total > 0 else 0.0
    positive_acc = (n_positive_correct / n_positive * 100) if n_positive > 0 else 0.0
    negative_acc = (n_negative_correct / n_negative * 100) if n_negative > 0 else 0.0
    
    tp = n_positive_correct
    fp = n_pred_true - tp
    fn = n_positive - tp
    
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Print summary
    summary = f"""
{'=' * 70}
EVALUATION SUMMARY
{'=' * 70}

Total samples evaluated: {n_total}
  - Ambiguous samples (ambiguity_type != 'no_ambiguity'): {n_positive}
  - Non-ambiguous samples (ambiguity_type == 'no_ambiguity'): {n_negative}

Predictions:
  - Predicted TRUE (ambiguous): {n_pred_true}
  - Predicted FALSE (not ambiguous): {n_pred_false}
  - Predicted NONE (parse error): {n_pred_none}

Accuracy:
  - Overall: {overall_acc:.1f}% ({n_correct}/{n_total})
  - Ambiguous samples: {positive_acc:.1f}% ({n_positive_correct}/{n_positive})
  - Non-ambiguous samples: {negative_acc:.1f}% ({n_negative_correct}/{n_negative})

Metrics for Ambiguous Detection:
  - Precision: {precision:.1f}%
  - Recall: {recall:.1f}%
  - F1 Score: {f1:.1f}%

Results saved to: {out_csv}
Checkpoint saved to: {checkpoint_path}
{'=' * 70}
"""
    log(summary)
    
    # Clean up
    if log_file:
        log_fp.close()
    
    # Remove checkpoint file after successful completion
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        log(f"Checkpoint file removed (evaluation complete)")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AREngine with checkpointing support"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON dataset file"
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Output CSV path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional)"
    )
    
    args = parser.parse_args()
    
    evaluate_with_checkpoints(
        args.json_path,
        out_csv=args.output_csv,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        log_file=args.log_file
    )


if __name__ == "__main__":
    main()