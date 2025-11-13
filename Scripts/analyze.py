#!/usr/bin/env python3
"""
Analyze evaluation results from AREngine ambiguity detection.

Usage:
    python analyze_results.py results.csv
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_results(csv_path):
    """
    Analyze evaluation results and provide detailed breakdown.
    
    Args:
        csv_path (str): Path to results CSV file
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("DETAILED RESULTS ANALYSIS")
    print("=" * 70)
    
    # Overall stats
    total = len(df)
    correct = df['correct'].sum()
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Overall accuracy: {accuracy:.1f}%")
    
    # Breakdown by sample type
    print(f"\n" + "-" * 70)
    print("Performance by Sample Type:")
    print("-" * 70)
    
    for sample_type in df['sample_type'].unique():
        subset = df[df['sample_type'] == sample_type]
        subset_correct = subset['correct'].sum()
        subset_total = len(subset)
        subset_acc = subset_correct / subset_total * 100 if subset_total > 0 else 0
        
        print(f"\n{sample_type.upper()}:")
        print(f"  Samples: {subset_total}")
        print(f"  Correct: {subset_correct}")
        print(f"  Accuracy: {subset_acc:.1f}%")
    
    # Breakdown by ambiguity type
    if 'ambiguity_type' in df.columns and not df['ambiguity_type'].isna().all():
        print(f"\n" + "-" * 70)
        print("Performance by Ambiguity Type:")
        print("-" * 70)
        
        for ambig_type in df['ambiguity_type'].unique():
            if pd.isna(ambig_type) or ambig_type == "":
                continue
            subset = df[df['ambiguity_type'] == ambig_type]
            subset_correct = subset['correct'].sum()
            subset_total = len(subset)
            subset_acc = subset_correct / subset_total * 100 if subset_total > 0 else 0
            
            print(f"\n{ambig_type}:")
            print(f"  Samples: {subset_total}")
            print(f"  Correct: {subset_correct}")
            print(f"  Accuracy: {subset_acc:.1f}%")
    
    # Confusion matrix
    print(f"\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    
    tp = len(df[(df['predicted_ambiguous'] == True) & (df['ground_truth_ambiguous'] == True)])
    fp = len(df[(df['predicted_ambiguous'] == True) & (df['ground_truth_ambiguous'] == False)])
    fn = len(df[(df['predicted_ambiguous'] == False) & (df['ground_truth_ambiguous'] == True)])
    tn = len(df[(df['predicted_ambiguous'] == False) & (df['ground_truth_ambiguous'] == False)])
    none_count = len(df[df['predicted_ambiguous'].isna()])
    
    print("\n                    Predicted")
    print("                 TRUE    FALSE   NONE")
    print(f"Actual  TRUE  | {tp:4d}    {fn:4d}    {none_count if none_count else 0:4d}")
    print(f"        FALSE | {fp:4d}    {tn:4d}")
    
    print(f"\nLegend:")
    print(f"  TP (True Positive): {tp} - Correctly identified ambiguous")
    print(f"  TN (True Negative): {tn} - Correctly identified not ambiguous")
    print(f"  FP (False Positive): {fp} - Incorrectly flagged as ambiguous")
    print(f"  FN (False Negative): {fn} - Missed ambiguity")
    if none_count:
        print(f"  NONE: {none_count} - Model output unparseable")
    
    # Error analysis
    print(f"\n" + "-" * 70)
    print("Error Analysis:")
    print("-" * 70)
    
    incorrect = df[df['correct'] == False]
    
    if len(incorrect) > 0:
        print(f"\nTotal errors: {len(incorrect)}")
        
        # False positives
        false_positives = df[(df['predicted_ambiguous'] == True) & (df['ground_truth_ambiguous'] == False)]
        print(f"\nFalse Positives (predicted ambiguous, but wasn't): {len(false_positives)}")
        if len(false_positives) > 0:
            print("\nExamples:")
            for idx, row in false_positives.head(3).iterrows():
                print(f"  - Scene {row['scene_id']}: '{row['initial_prompt'][:60]}...'")
                print(f"    Type: {row.get('ambiguity_type', 'N/A')}")
        
        # False negatives
        false_negatives = df[(df['predicted_ambiguous'] == False) & (df['ground_truth_ambiguous'] == True)]
        print(f"\nFalse Negatives (missed ambiguity): {len(false_negatives)}")
        if len(false_negatives) > 0:
            print("\nExamples:")
            for idx, row in false_negatives.head(3).iterrows():
                print(f"  - Scene {row['scene_id']}: '{row['initial_prompt'][:60]}...'")
                print(f"    Type: {row.get('ambiguity_type', 'N/A')}")
    else:
        print("\n🎉 Perfect accuracy! No errors found.")
    
    # Most common errors by type
    if 'ambiguity_type' in df.columns and len(incorrect) > 0:
        print(f"\n" + "-" * 70)
        print("Errors by Ambiguity Type:")
        print("-" * 70)
        
        error_by_type = incorrect.groupby('ambiguity_type').size().sort_values(ascending=False)
        total_by_type = df.groupby('ambiguity_type').size()
        
        for ambig_type, error_count in error_by_type.head(5).items():
            if pd.isna(ambig_type) or ambig_type == "":
                continue
            total_count = total_by_type[ambig_type]
            error_rate = error_count / total_count * 100
            print(f"\n{ambig_type}:")
            print(f"  Errors: {error_count}/{total_count} ({error_rate:.1f}%)")
    
    # Summary recommendations
    print(f"\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    
    if accuracy >= 90:
        print("\n✅ Excellent performance! System is working well.")
    elif accuracy >= 80:
        print("\n👍 Good performance. Minor improvements possible.")
    elif accuracy >= 70:
        print("\n⚠️ Moderate performance. Consider prompt engineering.")
    else:
        print("\n❌ Low performance. Significant improvements needed.")
    
    if fp > fn:
        print("\n📌 More false positives than false negatives.")
        print("   → System is too cautious (over-detects ambiguity)")
        print("   → Consider tightening ambiguity criteria in prompts")
    elif fn > fp:
        print("\n📌 More false negatives than false positives.")
        print("   → System misses some ambiguities")
        print("   → Consider loosening ambiguity criteria in prompts")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.csv>")
        print("\nExample:")
        print("  python analyze_results.py results.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: File '{csv_path}' not found")
        sys.exit(1)
    
    try:
        analyze_results(csv_path)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)