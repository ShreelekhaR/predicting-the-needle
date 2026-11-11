#!/usr/bin/env python3
"""
Build v8_weekly_clean.pt by replacing Spanner GT in v8.pt with clean weekly GTs.

For each positive location:
  1. Find weekly observations within ±14 days of completion
  2. Filter outliers via neighbor consistency
  3. Pick the candidate with highest text similarity to class description
  4. Require minimum text similarity (skip cloudy candidates)
  5. If no clean candidate found, fall back to Spanner GT

Negatives are unchanged (GT = before_emb).
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

DATA_DIR = "data"
CONSOLIDATED_DIR = os.path.join(DATA_DIR, "embeddings")
SEARCH_WINDOW_DAYS = 14
MIN_TEXT_SIM = 0.10
OUTLIER_THRESHOLD = 0.75


def parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def detect_outlier_dates(entries):
    if len(entries) < 3:
        return set()
    sorted_entries = sorted(entries, key=lambda e: e["date"])
    embs = [np.array(e["embedding"], dtype=np.float32) for e in sorted_entries]
    dates = [e["date"] for e in sorted_entries]

    neighbor_sims = []
    for i in range(len(embs)):
        sp = cosine_sim(embs[i], embs[i - 1]) if i > 0 else None
        sn = cosine_sim(embs[i], embs[i + 1]) if i < len(embs) - 1 else None
        neighbor_sims.append((dates[i], sp, sn))

    all_sims = [s for _, sp, sn in neighbor_sims for s in [sp, sn] if s is not None]
    if not all_sims:
        return set()

    median_sim = np.median(all_sims)
    threshold = max(OUTLIER_THRESHOLD, median_sim - 0.1)

    outliers = set()
    for date_str, sp, sn in neighbor_sims:
        if sp is None or sn is None:
            continue
        if sp < threshold and sn < threshold:
            outliers.add(date_str)
    return outliers


def find_clean_weekly_gt(weekly_list, completion_date, outlier_dates, text_emb):
    """Find cleanest weekly GT within ±SEARCH_WINDOW_DAYS of completion."""
    candidates = []
    for e in weekly_list:
        ds = e.get("date")
        if ds in outlier_dates:
            continue
        d = parse_date(ds)
        if d is None:
            continue
        diff = abs((d - completion_date).days)
        if diff <= SEARCH_WINDOW_DAYS:
            emb = np.array(e["embedding"], dtype=np.float32)
            sim = cosine_sim(emb, text_emb) if text_emb is not None else 0.0
            candidates.append((ds, emb, sim, diff))

    if not candidates:
        return None, None

    # Filter by minimum text similarity
    clean = [c for c in candidates if c[2] >= MIN_TEXT_SIM]
    if not clean:
        # Fall back to best available even if below threshold
        clean = candidates

    # Pick highest text similarity
    best = max(clean, key=lambda c: c[2])
    return best[1], best[2]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(DATA_DIR, "training_dataset_v8.pt"))
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "training_dataset_v8_weekly_clean.pt"))
    args = parser.parse_args()

    print(f"Loading dataset (Spanner GT): {args.input}")
    data = torch.load(args.input, weights_only=False, map_location="cpu")

    n_samples = len(data["class_labels"])
    print(f"  {n_samples} samples")

    print("Loading weekly embeddings...")
    with open(os.path.join(CONSOLIDATED_DIR, "ps_rssiglip_positive_weekly.json")) as f:
        pos_weekly = json.load(f)
    print(f"  {len(pos_weekly)} positive locations")

    print("Loading text embeddings...")
    with open(os.path.join(DATA_DIR, "text_rssiglip.json")) as f:
        text_data = json.load(f)
    text_class_embs = {}
    for cls_name, emb in zip(text_data["classes"], text_data["embeddings"]):
        text_class_embs[cls_name] = np.array(emb, dtype=np.float32)

    print("Loading completion dates...")
    hybrid_path = os.path.join(DATA_DIR, "completion_dates.json")
    with open(hybrid_path) as f:
        completion_dates = json.load(f)

    # Group sample indices by location_id
    loc_to_indices = defaultdict(list)
    for i in range(n_samples):
        loc_to_indices[data["location_ids"][i]].append(i)

    # Process each positive location
    gt_completion = data["gt_completion"].numpy().copy()
    gt_change = data["gt_change"].numpy().copy()
    is_negative = data["is_negative"]

    pos_locs = set()
    for loc_id, indices in loc_to_indices.items():
        if is_negative[indices[0]]:
            continue
        pos_locs.add(loc_id)

    print(f"\nProcessing {len(pos_locs)} positive locations...")
    stats = {"updated": 0, "kept_spanner": 0, "no_weekly": 0}
    text_sims = []

    for loc_id in sorted(pos_locs):
        indices = loc_to_indices[loc_id]
        cls = data["class_labels"][indices[0]]

        # Get weekly data
        weekly_list = pos_weekly.get(loc_id)
        if not weekly_list:
            stats["no_weekly"] += 1
            continue

        # Get completion date
        cd_info = completion_dates.get(loc_id, {})
        comp_dt = parse_date(cd_info.get("completion_date"))
        if comp_dt is None:
            stats["kept_spanner"] += 1
            continue

        # Get text embedding
        text_emb = text_class_embs.get(cls)

        # Find clean weekly GT
        outlier_dates = detect_outlier_dates(weekly_list)
        weekly_gt, text_sim = find_clean_weekly_gt(
            weekly_list, comp_dt, outlier_dates, text_emb
        )

        if weekly_gt is None:
            stats["kept_spanner"] += 1
            continue

        if text_sim is not None:
            text_sims.append(text_sim)

        # Update GT for all samples from this location
        before_emb = data["before_embs"][indices[0]].numpy()
        for idx in indices:
            gt_completion[idx] = weekly_gt
            gt_change[idx] = before_emb - weekly_gt

        stats["updated"] += 1

    print(f"\n  Updated: {stats['updated']} locations")
    print(f"  Kept Spanner (no clean weekly): {stats['kept_spanner']}")
    print(f"  No weekly data: {stats['no_weekly']}")

    if text_sims:
        sims = np.array(text_sims)
        print(f"\n  Weekly GT text similarity:")
        print(f"    mean={sims.mean():.4f}, median={np.median(sims):.4f}")
        print(f"    min={sims.min():.4f}, max={sims.max():.4f}")
        print(f"    >= 0.10: {100*(sims >= 0.10).mean():.1f}%")
        print(f"    >= 0.15: {100*(sims >= 0.15).mean():.1f}%")

    # Replace GT tensors
    data["gt_completion"] = torch.from_numpy(gt_completion)
    data["gt_change"] = torch.from_numpy(gt_change)

    print(f"\nSaving to {args.output}...")
    torch.save(data, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Done — {size_gb:.1f} GB")


if __name__ == "__main__":
    main()
