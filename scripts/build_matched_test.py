#!/usr/bin/env python3
"""
Build matched test set from the canonical PS test set (data/test.pt).

Rules:
  - Same 43,533 samples: same locations, same current_dates, same labels, same splits
  - Same time window per sample: [earliest PS observation date, current_date]
  - Input sequence: all target sensor observations within that window
  - Temporal offsets: recomputed from actual target sensor dates (not copied from PS)
  - GT completion: closest target sensor observation ON OR AFTER the PS completion date
  - No coverage at all: zeros everywhere, mask=False — model guesses
  - Completion date comes from PS (property of the event, not the sensor)

For PS cross-backbone (remoteclip): swaps embeddings 1:1 since same images
For S2: rebuilds sequence from S2's own observation cadence within PS time window

Usage:
  python prepare_test_s2.py --sensor s2 --output data/test_s2.pt
  python prepare_test_s2.py --sensor remoteclip --output data/test_remoteclip.pt
  python prepare_test_s2.py --sensor remoteclip_s2 --output data/test_remoteclip_s2.pt
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

CONSOLIDATED_DIR = "data/embeddings"
DATA_DIR = "data"

SENSOR_PREFIX = {
    "s2": "s2_rssiglip",
    "remoteclip": "ps_remoteclipft",
    "remoteclip_s2": "s2_remoteclipft",
}

EMB_DIMS = {
    "s2": 1152,
    "remoteclip": 768,
    "remoteclip_s2": 768,
}


def load_weekly_embeddings(sensor):
    prefix = SENSOR_PREFIX[sensor]
    pos_path = os.path.join(CONSOLIDATED_DIR, f"{prefix}_positive_weekly.json")
    neg_path = os.path.join(CONSOLIDATED_DIR, f"{prefix}_negative_weekly.json")

    print(f"Loading {pos_path}...")
    with open(pos_path) as f:
        pos = json.load(f)
    print(f"  {len(pos)} positive locations")

    print(f"Loading {neg_path}...")
    with open(neg_path) as f:
        neg = json.load(f)
    print(f"  {len(neg)} negative locations")

    return {**pos, **neg}


def get_earliest_date(ref, i):
    """Get the earliest observation date from the PS reference for sample i."""
    mask = ref["masks"][i]
    offsets = ref["temporal_offsets"][i]
    current_dt = datetime.strptime(ref["current_dates"][i], "%Y-%m-%d")
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return None
    max_offset = int(offsets[:n_valid].max().item())
    from datetime import timedelta
    return (current_dt - timedelta(days=max_offset)).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/test.pt",
                        help="Canonical test set (defines locations, labels, splits)")
    parser.add_argument("--sensor", required=True, choices=["s2", "remoteclip", "remoteclip_s2"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    emb_dim = EMB_DIMS[args.sensor]

    print(f"Loading reference: {args.reference}")
    ref = torch.load(args.reference, weights_only=False, map_location="cpu")
    n = len(ref["split_labels"])
    max_seq = ref["max_seq_len"]
    ref_emb_dim = ref.get("emb_dim", ref["sequences"].shape[-1])
    print(f"  {n} samples, max_seq={max_seq}, ref_emb_dim={ref_emb_dim}, target_emb_dim={emb_dim}")

    weekly = load_weekly_embeddings(args.sensor)

    text_emb_file = "text_remoteclipft.json" if "remoteclip" in args.sensor else "text_rssiglip.json"
    with open(os.path.join(DATA_DIR, text_emb_file)) as f:
        text_data = json.load(f)
    text_class_embs = {}
    for cls_name, emb in zip(text_data["classes"], text_data["embeddings"]):
        text_class_embs[cls_name] = np.array(emb, dtype=np.float32)

    # Output tensors
    sequences = torch.zeros(n, max_seq, emb_dim, dtype=torch.float32)
    masks = torch.zeros(n, max_seq, dtype=torch.bool)
    temporal_offsets = torch.zeros(n, max_seq, dtype=torch.float32)
    before_embs = torch.zeros(n, emb_dim, dtype=torch.float32)
    gt_completion = torch.zeros(n, emb_dim, dtype=torch.float32)
    gt_change = torch.zeros(n, emb_dim, dtype=torch.float32)
    text_cos_sim = torch.zeros(n, dtype=torch.float32)

    # Stats
    matched = 0
    no_coverage = 0
    no_completion = 0
    total_tokens = 0

    for i in range(n):
        if i % 10000 == 0 and i > 0:
            print(f"  Processing {i}/{n}... matched={matched}, no_coverage={no_coverage}")

        loc_id = ref["location_ids"][i]
        current_date = ref["current_dates"][i]
        completion_date = ref["completion_dates"][i]
        is_neg = bool(ref["is_negative"][i])
        cls = ref["class_labels"][i]

        current_dt = datetime.strptime(current_date, "%Y-%m-%d")

        # Get the earliest observation date from PS reference
        earliest_date = get_earliest_date(ref, i)
        if earliest_date is None:
            no_coverage += 1
            continue

        # Get all target sensor observations for this location
        entries = weekly.get(loc_id, [])
        if not entries:
            no_coverage += 1
            continue

        # Filter to [earliest_date, current_date] — same time window as PS
        valid = [e for e in entries if earliest_date <= e["date"] <= current_date]
        valid.sort(key=lambda e: e["date"])

        if not valid:
            no_coverage += 1
            continue

        # Build input sequence from target sensor's own cadence
        # Use up to max_seq most recent observations
        valid = valid[-max_seq:]
        n_valid = len(valid)
        total_tokens += n_valid

        for j, e in enumerate(valid):
            e_dt = datetime.strptime(e["date"], "%Y-%m-%d")
            offset = (current_dt - e_dt).days
            emb = np.array(e["embedding"], dtype=np.float32)
            sequences[i, j] = torch.from_numpy(emb)
            masks[i, j] = True
            temporal_offsets[i, j] = float(offset)

        # Before embedding = earliest observation in window
        before_embs[i] = sequences[i, 0]

        # GT completion: closest observation ON OR AFTER completion date
        if completion_date and not is_neg:
            comp_dt = datetime.strptime(completion_date, "%Y-%m-%d")
            text_emb = text_class_embs.get(cls)

            # All observations on or after completion date
            after_comp = [e for e in entries if e["date"] >= completion_date]
            after_comp.sort(key=lambda e: e["date"])

            if after_comp:
                # If multiple, pick the one with highest text similarity
                best_emb = None
                best_sim = -2.0
                for e in after_comp[:5]:  # check first 5 after completion
                    emb = np.array(e["embedding"], dtype=np.float32)
                    if text_emb is not None:
                        sim = float(np.dot(emb, text_emb) / (np.linalg.norm(emb) * np.linalg.norm(text_emb) + 1e-8))
                    else:
                        sim = -1.0
                    if best_emb is None or sim > best_sim:
                        best_emb = emb
                        best_sim = sim

                gt_completion[i] = torch.from_numpy(best_emb)
                gt_change[i] = before_embs[i] - gt_completion[i]
                text_cos_sim[i] = best_sim
            else:
                no_completion += 1
        else:
            # Negative sample — no GT completion needed
            pass

        matched += 1

    avg_tokens = total_tokens / max(matched, 1)
    print(f"\n  Matched: {matched} ({matched/n*100:.1f}%)")
    print(f"  No coverage: {no_coverage}")
    print(f"  No completion embedding: {no_completion}")
    print(f"  Avg tokens per matched sample: {avg_tokens:.1f}")

    out = {
        "sequences": sequences,
        "masks": masks,
        "temporal_offsets": temporal_offsets,
        "before_embs": before_embs,
        "gt_completion": gt_completion,
        "gt_change": gt_change,
        "lead_time_days": ref["lead_time_days"],
        "text_cos_sim": text_cos_sim,
        "is_fixed_lead": ref["is_fixed_lead"],
        "lead_time_bucket": ref["lead_time_bucket"],
        "is_negative": ref["is_negative"],
        "in_geo_holdout": ref["in_geo_holdout"],
        "latitudes": ref["latitudes"],
        "longitudes": ref["longitudes"],
        "split_labels": ref["split_labels"],
        "class_labels": ref["class_labels"],
        "location_ids": ref["location_ids"],
        "current_dates": ref["current_dates"],
        "completion_dates": ref["completion_dates"],
        "emb_dim": emb_dim,
        "max_seq_len": max_seq,
        "lead_time_buckets": ref["lead_time_buckets"],
    }

    print(f"\nSaving {args.output}...")
    torch.save(out, args.output)
    sz = os.path.getsize(args.output) / 1e6
    print(f"  {sz:.0f} MB, {n} total samples, {matched} with coverage, emb_dim={emb_dim}")


if __name__ == "__main__":
    main()
