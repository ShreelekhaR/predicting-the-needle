#!/usr/bin/env python3
"""
Build a matched dataset by swapping embeddings in the canonical V9 dataset.

Takes the canonical PS RS-SigLIP dataset (data/train.pt)
and replaces all embeddings with the target sensor's embeddings. Every sample
is preserved — same locations, lead times, splits. If a location has no
coverage in the target sensor, its embeddings are zeroed and mask is False.

Usage:
  python -m scripts.build_matched_dataset --sensor remoteclip --reference data/train.pt --output data/train_remoteclip.pt
  python -m scripts.build_matched_dataset --sensor s2 --reference data/train.pt --output data/train_s2.pt
  python -m scripts.build_matched_dataset --sensor remoteclip_s2 --reference data/train.pt --output data/train_remoteclip_s2.pt
"""

import argparse
import json
import os
from datetime import datetime, timedelta

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True,
                        help="Canonical dataset (e.g. data/train.pt)")
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

    # Load target sensor embeddings
    weekly = load_weekly_embeddings(args.sensor)

    # Load matching text embeddings for GT quality scoring
    text_emb_file = "text_remoteclipft.json" if "remoteclip" in args.sensor else "text_rssiglip.json"
    with open(os.path.join(DATA_DIR, text_emb_file)) as f:
        text_data = json.load(f)
    text_class_embs = {}
    for cls_name, emb in zip(text_data["classes"], text_data["embeddings"]):
        text_class_embs[cls_name] = np.array(emb, dtype=np.float32)

    # Build lookup: for each location, sorted list of (date, embedding)
    weekly_lookup = {}
    for loc_id, entries in weekly.items():
        sorted_entries = sorted(entries, key=lambda e: e["date"])
        weekly_lookup[loc_id] = sorted_entries

    # New tensors
    sequences = torch.zeros(n, max_seq, emb_dim, dtype=torch.float32)
    masks = torch.zeros(n, max_seq, dtype=torch.bool)
    temporal_offsets = ref["temporal_offsets"].clone()
    before_embs = torch.zeros(n, emb_dim, dtype=torch.float32)
    gt_completion = torch.zeros(n, emb_dim, dtype=torch.float32)
    gt_change = torch.zeros(n, emb_dim, dtype=torch.float32)
    text_cos_sim = torch.zeros(n, dtype=torch.float32)

    # Stats
    matched = 0
    no_coverage = 0
    no_completion = 0
    partial = 0

    for i in range(n):
        if i % 10000 == 0:
            print(f"  Processing {i}/{n}...")

        loc_id = ref["location_ids"][i]
        current_date = ref["current_dates"][i]
        completion_date = ref["completion_dates"][i]
        is_neg = bool(ref["is_negative"][i])
        cls = ref["class_labels"][i]

        entries = weekly_lookup.get(loc_id, [])

        if not entries:
            no_coverage += 1
            continue

        current_dt = datetime.strptime(current_date, "%Y-%m-%d")

        # Build input sequence: all observations before current_date
        valid = [e for e in entries if e["date"] <= current_date]
        if not valid:
            no_coverage += 1
            continue

        # Use the same temporal structure as reference
        # Match by: for each reference token position that has mask=True,
        # find the closest target sensor observation
        ref_mask = ref["masks"][i]
        ref_offsets = ref["temporal_offsets"][i]
        n_ref_valid = int(ref_mask.sum().item())

        if n_ref_valid == 0:
            no_coverage += 1
            continue

        # Strategy: match by exact date (for same-sensor cross-backbone like PS RemoteCLIP)
        # Build date lookup for fast matching
        date_to_emb = {e["date"]: e for e in valid}

        n_filled = 0
        for j in range(n_ref_valid):
            offset_days = int(ref_offsets[j].item())
            target_date = (current_dt - timedelta(days=offset_days)).strftime("%Y-%m-%d")

            # Exact date match first
            best_entry = date_to_emb.get(target_date)
            best_dist = 0 if best_entry else float("inf")

            # Fallback: closest within 3 days (for minor date discrepancies)
            if best_entry is None:
                for e in valid:
                    dist = abs((datetime.strptime(e["date"], "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")).days)
                    if dist < best_dist:
                        best_dist = dist
                        best_entry = e

            if best_entry is not None and best_dist <= 3:
                emb = np.array(best_entry["embedding"], dtype=np.float32)
                sequences[i, j] = torch.from_numpy(emb)
                masks[i, j] = True
                n_filled += 1

        if n_filled == 0:
            no_coverage += 1
            continue

        if n_filled < n_ref_valid:
            partial += 1

        # Before embedding = first valid observation
        for j in range(max_seq):
            if masks[i, j]:
                before_embs[i] = sequences[i, j]
                break

        # GT completion embedding
        if completion_date and not is_neg:
            comp_dt = datetime.strptime(completion_date, "%Y-%m-%d")
            search_days = 21
            best_emb = None
            best_sim = -2.0
            text_emb = text_class_embs.get(cls)

            for e in entries:
                e_dt = datetime.strptime(e["date"], "%Y-%m-%d")
                dist = abs((e_dt - comp_dt).days)
                if dist <= search_days:
                    emb = np.array(e["embedding"], dtype=np.float32)
                    if text_emb is not None:
                        sim = float(np.dot(emb, text_emb) / (np.linalg.norm(emb) * np.linalg.norm(text_emb) + 1e-8))
                    else:
                        sim = -1.0
                    if best_emb is None or sim > best_sim:
                        best_emb = emb
                        best_sim = sim

            if best_emb is not None:
                gt_completion[i] = torch.from_numpy(best_emb)
                gt_change[i] = before_embs[i] - gt_completion[i]
                text_cos_sim[i] = best_sim
            else:
                no_completion += 1

        matched += 1

    print(f"\n  Matched: {matched}")
    print(f"  No coverage: {no_coverage}")
    print(f"  Partial coverage: {partial}")
    print(f"  No completion embedding: {no_completion}")

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
    print(f"  {sz:.0f} MB, {n} samples, {matched} with coverage, emb_dim={emb_dim}")


if __name__ == "__main__":
    main()
