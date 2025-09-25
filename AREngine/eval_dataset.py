import json
import csv
from pathlib import Path
from tqdm.auto import tqdm

from arengine import AREngine

def evaluate_json_file(json_path, out_csv="results.csv", limit=None):
    data = json.loads(Path(json_path).read_text())
    
    # Respect an optional limit while keeping tqdm's total correct
    items = data if limit is None else data[:limit]
    total = len(items)
    
    engine = AREngine()
    engine.load_model()

    results = []
    n_total = 0
    n_pred_true = 0
    n_pred_false = 0
    n_pred_none = 0

    for i, item in enumerate(
        tqdm(
            items,
            total=total,
            desc="Evaluating",
            unit="scene",
            dynamic_ncols=True,
        )
    ):
        if limit is not None and i >= limit:
            break

        scene = {"environment_info": (item.get("environment_info") or "").strip()}
        prompt = item.get("dialogue", [])[0]["text"]

        pred, raw = engine.is_prompt_ambiguous(prompt, scene=scene)
        # ground truth is always ambiguous (True)
        gt = True

        n_total += 1
        if pred is True:
            n_pred_true += 1
        elif pred is False:
            n_pred_false += 1
        else:
            n_pred_none += 1

        results.append({
            "scene_id": item.get("scene_id"),
            "object_id": item.get("object_id"),
            "object_name": item.get("object_name"),
            "initial_prompt": prompt,
            "predicted_ambiguous": pred,
            "raw_model_output": raw,
            "ground_truth_ambiguous": gt,
            "correct": (pred is True),  # since gt=True for all items
        })

    # write a simple CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "object_id",
                "object_name",
                "initial_prompt",
                "predicted_ambiguous",
                "raw_model_output",
                "ground_truth_ambiguous",
                "correct",
            ],
        )
        w.writeheader()
        w.writerows(results)

    # print summary
    acc = (n_pred_true / n_total) if n_total else 0.0
    print(f"Total: {n_total}")
    print(f"Pred TRUE: {n_pred_true}  | Pred FALSE: {n_pred_false}  | Pred NONE: {n_pred_none}")
    print(f"Accuracy vs GT (all True): {acc:.3f}")
    print(f"Wrote: {out_csv}")

# Example: python eval_dataset.py /path/to/scenes.json results.csv
if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else "ambiguity_dataset_demo_10.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    evaluate_json_file(in_path, out_csv=out_path, limit=None)
