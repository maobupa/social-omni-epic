#!/usr/bin/env python3
"""Validate coherence checker and LP judge against a human-labeled set (§10.4).

Usage:
  python scripts/validate_judges.py \\
      --labels data/judge_labels.jsonl \\
      --judge-model google/gemini-3-flash-preview \\
      --learner-model openai/gpt-5-mini

Label file format (JSONL, one record per line):
  {
    "scenario": { ...SocialScenario fields... },
    "transcripts": [ [{"turn":..,"speaker":..,"content":..}, ...], ... ],
    "lp_label": 0.0,          // ground-truth LP (human-judged improved_votes/total_votes)
    "coherent": true,          // ground-truth coherence verdict
    "notes": "optional"
  }

Outputs a summary table: precision/recall/accuracy for coherence; MAE/correlation for LP.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path


def _load_labels(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _build_scenario(d: dict):
    from social_omni_epic.data_models import SocialScenario
    try:
        return SocialScenario(**d)
    except Exception as e:
        print(f"[warn] could not build SocialScenario: {e}", file=sys.stderr)
        return None


async def _run_lp(fm_judge, label: dict):
    from social_omni_epic.lp_judge import compute_lp
    from social_omni_epic.data_models import SocialScenario

    scn = _build_scenario(label.get("scenario", {}))
    if scn is None:
        return None
    transcripts = label.get("transcripts", [])
    learner_goal = label.get("learner_goal", "")
    relational_stakes = label.get("relational_stakes", "")
    try:
        result = await compute_lp(
            fm_judge=fm_judge,
            scenario=scn,
            transcripts=transcripts,
            learner_goal=learner_goal,
            relational_stakes=relational_stakes,
        )
        return result.lp_value
    except Exception as e:
        print(f"[warn] LP judge error: {e}", file=sys.stderr)
        return None


def _run_coherence(coherence_checker, label: dict):
    scn = _build_scenario(label.get("scenario", {}))
    if scn is None:
        return None
    result = coherence_checker.check(scn)
    return result.passed


async def main_async(args):
    labels = _load_labels(args.labels)
    print(f"Loaded {len(labels)} labeled examples")

    from social_omni_epic.fm import FM
    from social_omni_epic.coherence_check import CoherenceChecker

    fm_judge = FM(model=args.judge_model)
    coherence_checker = CoherenceChecker(fm=FM(model=args.judge_model))

    coherence_preds, coherence_labels = [], []
    lp_preds, lp_labels = [], []

    for i, label in enumerate(labels):
        print(f"  [{i+1}/{len(labels)}] ", end="", flush=True)

        if "coherent" in label:
            pred = _run_coherence(coherence_checker, label)
            if pred is not None:
                coherence_preds.append(pred)
                coherence_labels.append(bool(label["coherent"]))
                print(f"coherence={'PASS' if pred else 'FAIL'} (label={'PASS' if label['coherent'] else 'FAIL'})", end="")
            else:
                print("coherence=ERROR", end="")

        if "lp_label" in label and label.get("transcripts"):
            pred_lp = await _run_lp(fm_judge, label)
            if pred_lp is not None:
                lp_preds.append(pred_lp)
                lp_labels.append(float(label["lp_label"]))
                print(f"  LP={pred_lp:.2f} (label={label['lp_label']:.2f})", end="")
            else:
                print("  LP=ERROR", end="")

        print()

    print("\n=== Validation Results ===")

    if coherence_preds:
        n = len(coherence_preds)
        correct = sum(p == l for p, l in zip(coherence_preds, coherence_labels))
        tp = sum(p and l for p, l in zip(coherence_preds, coherence_labels))
        fp = sum(p and not l for p, l in zip(coherence_preds, coherence_labels))
        fn = sum(not p and l for p, l in zip(coherence_preds, coherence_labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"Coherence  n={n}  accuracy={correct/n:.2%}  precision={precision:.2%}  recall={recall:.2%}")

    if lp_preds:
        import numpy as np
        n = len(lp_preds)
        preds_arr = np.array(lp_preds)
        labels_arr = np.array(lp_labels)
        mae = float(np.mean(np.abs(preds_arr - labels_arr)))
        if np.std(preds_arr) > 0 and np.std(labels_arr) > 0:
            corr = float(np.corrcoef(preds_arr, labels_arr)[0, 1])
        else:
            corr = float("nan")
        print(f"LP judge   n={n}  MAE={mae:.3f}  Pearson r={corr:.3f}")

    if args.output:
        report = {
            "coherence": {"n": len(coherence_preds), "predictions": coherence_preds, "labels": coherence_labels},
            "lp": {"n": len(lp_preds), "predictions": lp_preds, "labels": lp_labels},
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Validate coherence and LP judges")
    parser.add_argument("--labels", required=True, help="JSONL file with labeled examples")
    parser.add_argument("--judge-model", default="openai/gpt-5-mini")
    parser.add_argument("--output", default=None, help="Write JSON report here")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
