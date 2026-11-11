#!/usr/bin/env python3
"""
Evaluate time-to-completion predictions with equal sampling across lead time buckets.

Verifies the time prediction head works across all horizons, not just dominated
by one time range. Reports per-bucket statistics and generates a summary table.

Usage:
  python -m scripts.eval_time_to_completion --model results_v9_weekly_time/B_learn_after_diffs/epoch_120.pt --dataset data/test.pt
  python -m scripts.eval_time_to_completion --model results_remoteclip_v2/B_learn_after_diffs/best_model.pt --dataset data/test_remoteclip.pt --text-emb data/text_remoteclipft.json
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anticipation.models.autoregressive_v4 import TemporalConditionedAR
from anticipation.models.dataset_v5 import AnticipationDatasetV5


LEAD_BUCKETS = [
    ("1w-1mo",   7,   30),
    ("1-3mo",   30,   90),
    ("3-6mo",   90,  180),
    ("6-9mo",  180,  270),
    ("9mo+",   270, 9999),
]

N_PER_BUCKET = 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-per-bucket", type=int, default=N_PER_BUCKET)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--diffs", action="store_true")
    args = parser.parse_args()

    # Architecture comes from the checkpoint itself (release file names carry no experiment name):
    # velocity on iff it has diff_encoder weights; learned input positions iff it has that table.
    state = torch.load(args.model, weights_only=True, map_location="cpu")
    use_diffs = args.diffs or any(k.startswith("diff_encoder.") for k in state)
    use_sinusoidal = not any(k.startswith("pos_embedding.") for k in state)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else torch.device("mps") if torch.backends.mps.is_available()
                          else "cpu")

    print("Loading dataset...")
    data = torch.load(args.dataset, weights_only=False, map_location="cpu")
    n = len(data["split_labels"])
    emb_dim = data.get("emb_dim", data["sequences"].shape[-1])

    print("Loading model...")
    model = TemporalConditionedAR(
        emb_dim=emb_dim, n_heads=8, n_layers=4, ff_dim=2048,
        dropout=0.0, max_seq_len=data["max_seq_len"],
        use_diffs=use_diffs, use_learned_input_pos=not use_sinusoidal,
    ).to(device)
    model.load_state_dict(state, strict=True)
    print(f"  velocity (diffs): {use_diffs} | learned input positions: {not use_sinusoidal}")
    model.eval()

    # Collect positive test/oov samples with lead times
    print("Collecting samples...")
    all_samples = []
    for i in range(n):
        if data["is_negative"][i]:
            continue
        lead = float(data["lead_time_days"][i].item())
        if lead <= 0:
            continue
        n_tok = int(data["masks"][i].sum().item())
        if n_tok < 2:
            continue
        all_samples.append({
            "idx": i,
            "lead_days": lead,
            "class": data["class_labels"][i],
            "split": data["split_labels"][i],
        })

    print(f"  {len(all_samples)} positive samples with lead > 0")

    # Equal sampling per bucket
    np.random.seed(42)
    sampled = []
    for bname, bmin, bmax in LEAD_BUCKETS:
        bucket = [s for s in all_samples if bmin <= s["lead_days"] < bmax]
        n_take = min(args.n_per_bucket, len(bucket))
        if n_take > 0:
            chosen = np.random.choice(len(bucket), size=n_take, replace=False)
            for j in chosen:
                sampled.append(bucket[j])
        print(f"  {bname:8s}: {len(bucket):5d} available, {n_take} sampled")

    print(f"  Total sampled: {len(sampled)}")

    # Run inference on sampled indices
    indices = [s["idx"] for s in sampled]
    gt_leads = np.array([s["lead_days"] for s in sampled])
    classes = [s["class"] for s in sampled]

    pred_leads = np.zeros(len(indices))
    bs = 64

    with torch.no_grad():
        for start in range(0, len(indices), bs):
            end = min(start + bs, len(indices))
            batch_idx = indices[start:end]

            seq = data["sequences"][batch_idx].to(device)
            mask = data["masks"][batch_idx].to(device)
            before = data["before_embs"][batch_idx].to(device)
            offsets = data["temporal_offsets"][batch_idx].to(device)
            query = data["lead_time_days"][batch_idx].to(device)
            # Location is a model input (GeographicEncoding); without it predictions are made
            # at an unseen location. scripts/evaluate.py passes it too.
            lat = data["latitudes"][batch_idx].to(device)
            lon = data["longitudes"][batch_idx].to(device)

            pred, time_pred = model(seq, mask, before,
                                    temporal_offsets=offsets,
                                    query_offset=query,
                                    latitude=lat, longitude=lon,
                                    predict_time=True)

            pred_leads[start:end] = (time_pred.cpu().numpy().flatten() * 365)

    # Overall stats
    errors = np.abs(gt_leads - pred_leads)
    corr = np.corrcoef(gt_leads, pred_leads)[0, 1]

    print(f"\n{'='*70}")
    print(f"TIME-TO-COMPLETION EVALUATION (equally sampled)")
    print(f"{'='*70}")
    print(f"  N = {len(sampled)}")
    print(f"  Correlation: r = {corr:.3f}")
    print(f"  Mean abs error: {errors.mean():.0f}d")
    print(f"  Median abs error: {np.median(errors):.0f}d")
    print(f"  Within 7d:  {(errors <= 7).mean()*100:.1f}%")
    print(f"  Within 14d: {(errors <= 14).mean()*100:.1f}%")
    print(f"  Within 30d: {(errors <= 30).mean()*100:.1f}%")
    print(f"  Within 60d: {(errors <= 60).mean()*100:.1f}%")

    # Per-bucket stats
    print(f"\n  {'Bucket':8s} | {'N':>5s} | {'r':>6s} | {'Mean':>6s} | {'Med':>5s} | {'≤7d':>5s} | {'≤14d':>5s} | {'≤30d':>5s} | {'≤60d':>5s}")
    print(f"  {'-'*75}")

    bucket_results = []
    for bname, bmin, bmax in LEAD_BUCKETS:
        bmask = (gt_leads >= bmin) & (gt_leads < bmax)
        if bmask.sum() == 0:
            continue
        b_gt = gt_leads[bmask]
        b_pred = pred_leads[bmask]
        b_err = np.abs(b_gt - b_pred)
        b_corr = np.corrcoef(b_gt, b_pred)[0, 1] if bmask.sum() > 1 else 0

        print(f"  {bname:8s} | {bmask.sum():5d} | {b_corr:6.3f} | {b_err.mean():5.0f}d | {np.median(b_err):4.0f}d | "
              f"{(b_err<=7).mean()*100:4.1f}% | {(b_err<=14).mean()*100:4.1f}% | "
              f"{(b_err<=30).mean()*100:4.1f}% | {(b_err<=60).mean()*100:4.1f}%")

        bucket_results.append({
            "bucket": bname, "n": int(bmask.sum()), "r": b_corr,
            "mean": b_err.mean(), "median": np.median(b_err),
            "within_7d": (b_err <= 7).mean(), "within_30d": (b_err <= 30).mean(),
        })

    # Scatter plot
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter with color by bucket
    ax = axes[0]
    colors = {'1w-1mo': '#2ca02c', '1-3mo': '#1f77b4', '3-6mo': '#ff7f0e', '6-9mo': '#d62728', '9mo+': '#9467bd'}
    for bname, bmin, bmax in LEAD_BUCKETS:
        bmask = (gt_leads >= bmin) & (gt_leads < bmax)
        if bmask.sum() == 0:
            continue
        ax.scatter(gt_leads[bmask], pred_leads[bmask], s=8, alpha=0.4,
                   c=colors.get(bname, '#333'), label=bname, rasterized=True)
    maxv = max(gt_leads.max(), pred_leads.max()) * 1.05
    ax.plot([0, maxv], [0, maxv], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel("Ground Truth (days)", fontsize=11)
    ax.set_ylabel("Predicted (days)", fontsize=11)
    ax.set_title(f"Time-to-Completion (r={corr:.3f}, median={np.median(errors):.0f}d)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.set_aspect("equal")

    # Right: error distribution per bucket
    ax2 = axes[1]
    bucket_names = [b[0] for b in LEAD_BUCKETS]
    bucket_medians = []
    bucket_q25 = []
    bucket_q75 = []
    for bname, bmin, bmax in LEAD_BUCKETS:
        bmask = (gt_leads >= bmin) & (gt_leads < bmax)
        if bmask.sum() == 0:
            bucket_medians.append(0)
            bucket_q25.append(0)
            bucket_q75.append(0)
            continue
        b_err = np.abs(gt_leads[bmask] - pred_leads[bmask])
        bucket_medians.append(np.median(b_err))
        bucket_q25.append(np.percentile(b_err, 25))
        bucket_q75.append(np.percentile(b_err, 75))

    x = np.arange(len(bucket_names))
    ax2.bar(x, bucket_medians, color=[colors.get(b, '#333') for b in bucket_names], alpha=0.7)
    ax2.errorbar(x, bucket_medians,
                 yerr=[np.array(bucket_medians) - np.array(bucket_q25),
                       np.array(bucket_q75) - np.array(bucket_medians)],
                 fmt='none', ecolor='black', capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_names)
    ax2.set_ylabel("Median Absolute Error (days)", fontsize=11)
    ax2.set_title("Error by Lead Time Bucket", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "time_prediction_eval.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved plot to {out_path}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
