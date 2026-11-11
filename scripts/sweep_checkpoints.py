#!/usr/bin/env python3
"""
Sweep checkpoints through scripts/evaluate.py full eval (Table 2 only).

Copies each checkpoint as best_model.pt, runs scripts/evaluate.py, collects results.
Uses scripts/evaluate.py's exact evaluation logic — no reimplementation.

Usage:
  python -m scripts.sweep_checkpoints --results-dir results_v9_gated --experiment B_learn_after_gated_diffs --dataset data/test.pt
"""

import argparse
import glob
import json
import os
import shutil
import sys
import subprocess


def run_eval(results_dir, experiment, dataset, text_emb, require_gpu=False):
    """Run scripts/evaluate.py and capture JSON output."""
    cmd = [
        sys.executable, "-m", "scripts.evaluate",
        "--results-dir", results_dir,
        "--dataset", dataset,
        "--text-emb", text_emb,
        "--experiment", experiment,
    ]
    if require_gpu:
        cmd.append("--require-gpu")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Parent results dir (e.g. results_v9_gated)")
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (e.g. B_learn_after_gated_diffs)")
    parser.add_argument("--dataset", default="data/test.pt")
    parser.add_argument("--text-emb", default="data/text_rssiglip.json")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Exit if GPU is not available (prevents CPU OOM)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Skip checkpoints before this epoch")
    args = parser.parse_args()

    exp_dir = os.path.join(args.results_dir, args.experiment)

    # For scripts/evaluate.py to find the experiment, it needs to be a known name.
    # We'll use a temp alias if needed.
    from scripts.train import EXPERIMENTS as V5_EXPERIMENTS
    known_exp = args.experiment in V5_EXPERIMENTS

    # If unknown experiment, create a temp alias as B_learn_after
    alias = args.experiment if known_exp else "B_learn_after"
    alias_dir = os.path.join(args.results_dir, alias)
    using_alias = not known_exp and alias != args.experiment

    if using_alias:
        if os.path.exists(alias_dir):
            print(f"ERROR: {alias_dir} already exists, can't create alias")
            return
        os.symlink(os.path.abspath(exp_dir), alias_dir)
        print(f"  Created symlink: {alias} -> {args.experiment}")

    try:
        # Find all checkpoints
        ckpts = sorted(glob.glob(os.path.join(exp_dir, "epoch_*.pt")),
                       key=lambda p: int(os.path.basename(p).replace("epoch_", "").replace(".pt", "")))
        if args.start_epoch > 0:
            ckpts = [p for p in ckpts if int(os.path.basename(p).replace("epoch_", "").replace(".pt", "")) >= args.start_epoch]
        for special in ["best_model.pt", "best_seen.pt", "best_oov.pt"]:
            p = os.path.join(exp_dir, special)
            if os.path.exists(p) and p not in ckpts:
                ckpts.append(p)

        if not ckpts:
            print(f"No checkpoints in {exp_dir}")
            return

        best_model_path = os.path.join(exp_dir, "best_model.pt")
        original_best = None
        if os.path.exists(best_model_path):
            original_best = best_model_path + ".sweep_backup"
            shutil.copy2(best_model_path, original_best)

        print(f"Sweeping {len(ckpts)} checkpoints through scripts/evaluate.py (full eval)")
        print(f"  Experiment: {args.experiment} (alias: {alias})")
        print(f"  Dataset: {args.dataset}")

        # Header
        hdr = f"{'Checkpoint':>25s} | {'Seen':>6s} {'m@10':>5s} | {'OOV':>6s} {'m@10':>5s} | {'GeoSn':>6s} {'m@10':>5s} | {'GeoOV':>6s} {'m@10':>5s} | {'F1':>5s}"
        print("\n" + hdr)
        print("-" * len(hdr))

        all_results = []
        best = {"seen": (-1, ""), "oov": (-1, ""), "geo_oov": (-1, "")}

        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)

            # Swap checkpoint as best_model.pt
            if os.path.abspath(ckpt_path) != os.path.abspath(best_model_path):
                shutil.copy2(ckpt_path, best_model_path)

            # Run eval
            stdout, rc = run_eval(args.results_dir, alias, args.dataset, args.text_emb, require_gpu=args.require_gpu)
            if rc != 0:
                print(f"{ckpt_name:>25s} | ERROR")
                continue

            # Parse eval JSON
            eval_json = os.path.join(args.results_dir, "eval_v5.json")
            if not os.path.exists(eval_json):
                print(f"{ckpt_name:>25s} | NO JSON")
                continue

            with open(eval_json) as f:
                eval_data = json.load(f)

            # Extract metrics from the full eval (not fixed)
            exp_key = f"{alias}_all" if f"{alias}_all" in eval_data else alias
            if exp_key not in eval_data:
                # Try any key
                exp_key = [k for k in eval_data if not k.endswith("_fixed")][0] if eval_data else None
            if not exp_key:
                print(f"{ckpt_name:>25s} | NO DATA")
                continue

            settings = eval_data[exp_key].get("settings", {})
            seen = settings.get("Seen", {})
            oov = settings.get("OOV", {})
            geo_seen = settings.get("Geo Seen", {})
            geo_oov = settings.get("Geo OOV", {})

            s5 = seen.get("Model", {}).get(5, seen.get("Model", {}).get("5", 0))
            s10 = seen.get("Model", {}).get(10, seen.get("Model", {}).get("10", 0))
            o5 = oov.get("Model", {}).get(5, oov.get("Model", {}).get("5", 0))
            o10 = oov.get("Model", {}).get(10, oov.get("Model", {}).get("10", 0))
            gs5 = geo_seen.get("Model", {}).get(5, geo_seen.get("Model", {}).get("5", 0))
            gs10 = geo_seen.get("Model", {}).get(10, geo_seen.get("Model", {}).get("10", 0))
            go5 = geo_oov.get("Model", {}).get(5, geo_oov.get("Model", {}).get("5", 0))
            go10 = geo_oov.get("Model", {}).get(10, geo_oov.get("Model", {}).get("10", 0))
            f1 = seen.get("det_f1", 0)

            print(f"{ckpt_name:>25s} | {s5:6.3f} {s10:5.3f} | {o5:6.3f} {o10:5.3f} | {gs5:6.3f} {gs10:5.3f} | {go5:6.3f} {go10:5.3f} | {f1:5.3f}")

            row = {"ckpt": ckpt_name, "seen_mAP5": s5, "seen_mAP10": s10,
                   "oov_mAP5": o5, "oov_mAP10": o10,
                   "geo_seen_mAP5": gs5, "geo_seen_mAP10": gs10,
                   "geo_oov_mAP5": go5, "geo_oov_mAP10": go10, "f1": f1}
            all_results.append(row)

            if s10 > best["seen"][0]: best["seen"] = (s10, ckpt_name)
            if o10 > best["oov"][0]: best["oov"] = (o10, ckpt_name)
            if go10 > best["geo_oov"][0]: best["geo_oov"] = (go10, ckpt_name)

        print("-" * len(hdr))
        print(f"\n  Best Seen mAP@10:    {best['seen'][0]:.3f} ({best['seen'][1]})")
        print(f"  Best OOV mAP@10:     {best['oov'][0]:.3f} ({best['oov'][1]})")
        print(f"  Best Geo OOV mAP@10: {best['geo_oov'][0]:.3f} ({best['geo_oov'][1]})")

        # Save
        out_path = os.path.join(exp_dir, "sweep_results.json")
        with open(out_path, "w") as f:
            json.dump({"results": all_results, "best": best}, f, indent=2)
        print(f"\n  Saved {out_path}")

        # Restore original best_model.pt
        if original_best and os.path.exists(original_best):
            shutil.move(original_best, best_model_path)

    finally:
        # Clean up symlink
        if using_alias and os.path.islink(alias_dir):
            os.unlink(alias_dir)


if __name__ == "__main__":
    main()
