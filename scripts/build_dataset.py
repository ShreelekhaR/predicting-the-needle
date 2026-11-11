#!/usr/bin/env python3
"""
Prepare V5 training dataset: completion-aligned GT for satellite event anticipation.

V5 key change from V4:
  - Single GT per location = completion embedding (first observation where event
    looks more like after_description than before_description)
  - Lead time as sampling axis: samples at 1wk/1mo/3mo/6mo before completion + random
  - GT verified against class text embedding (text_cos_sim stored per sample)
  - Arch B preserved: model takes lead_time_days as query_offset

Usage:
  python prepare_dataset_v5.py
  python prepare_dataset_v5.py --min-text-sim 0.1 --n-random 5
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import torch

from scripts.completion_detection import (
    build_m2_series, detect_permanent, detect_outliers as td_detect_outliers
)

DATA_DIR = "data"
CONSOLIDATED_DIR = os.path.join(DATA_DIR, "embeddings")

LEAD_TIMES = {"1mo": 30, "3mo": 90, "6mo": 180, "9mo": 270}
N_RANDOM_LEAD = 5
MAX_WEEKLY = 53
EMB_DIM = 1152
OUTLIER_THRESHOLD = 0.75
COMPLETION_SEARCH_DAYS = 7  # search ±7 days around completion date


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
    """Flag cloudy/noisy images via temporal neighbor consistency."""
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


def find_completion_embedding(weekly_list, completion_date, outlier_dates, text_emb=None):
    """Find best completion embedding within ±COMPLETION_SEARCH_DAYS.

    If text_emb is provided, picks the candidate with highest cosine_sim
    to the class text embedding (clearest, most representative image).
    Otherwise picks the closest date.

    Returns (embedding, matched_date, text_sim) or (None, None, None).
    """
    candidates = []
    for e in weekly_list:
        ds = e.get("date")
        if ds in outlier_dates:
            continue
        d = parse_date(ds)
        if d is None:
            continue
        diff = abs((d - completion_date).days)
        if diff <= COMPLETION_SEARCH_DAYS:
            emb = np.array(e["embedding"], dtype=np.float32)
            candidates.append((ds, d, emb, diff))

    if not candidates:
        return None, None, None

    if text_emb is not None and len(candidates) > 1:
        # Pick candidate with highest text similarity
        best = max(candidates, key=lambda c: cosine_sim(c[2], text_emb))
    else:
        # Pick closest date
        best = min(candidates, key=lambda c: c[3])

    ds, d, emb, diff = best
    sim = cosine_sim(emb, text_emb) if text_emb is not None else -1.0
    return emb, ds, sim


def gather_weekly_sequence(weekly_list, current_date, outlier_dates):
    """Build input: all clean weekly embeddings before current_date, up to MAX_WEEKLY."""
    tokens = []
    for e in weekly_list:
        ds = e.get("date")
        d = parse_date(ds)
        if d is None or d >= current_date or ds in outlier_dates:
            continue
        tokens.append((d, np.array(e["embedding"], dtype=np.float32)))

    tokens.sort(key=lambda x: x[0])
    tokens = tokens[-MAX_WEEKLY:]

    if not tokens:
        return None

    n = len(tokens)
    padded_seq = np.zeros((MAX_WEEKLY, EMB_DIM), dtype=np.float32)
    mask = np.zeros(MAX_WEEKLY, dtype=np.bool_)
    temporal_offsets = np.full(MAX_WEEKLY, -1.0, dtype=np.float32)

    for i, (dt, emb) in enumerate(tokens):
        padded_seq[i] = emb
        mask[i] = True
        temporal_offsets[i] = (current_date - dt).days

    return {
        "sequence": padded_seq,
        "mask": mask,
        "temporal_offsets": temporal_offsets,
        "n_tokens": n,
    }


def generate_fixed_lead_samples(weekly_list, completion_dt, outlier_dates):
    """Generate samples at fixed lead times [7, 30, 90, 180] days before completion."""
    results = []
    for bucket_idx, (name, days) in enumerate(LEAD_TIMES.items()):
        current_dt = completion_dt - timedelta(days=days)

        seq = gather_weekly_sequence(weekly_list, current_dt, outlier_dates)
        if seq is None:
            continue

        results.append({
            **seq,
            "current_date": current_dt.strftime("%Y-%m-%d"),
            "lead_time_days": days,
            "is_fixed_lead": True,
            "lead_time_bucket": bucket_idx,
        })
    return results


def generate_random_lead_samples(weekly_list, completion_dt, outlier_dates, n=N_RANDOM_LEAD):
    """Generate samples at random lead times before completion.

    Random dates are drawn from the range [first_observation + 7 days, completion - 7 days]
    to ensure meaningful input sequences and avoid overlap with the 1wk fixed lead.
    """
    # Find date range of available observations
    obs_dates = sorted(
        parse_date(e["date"]) for e in weekly_list
        if parse_date(e.get("date")) and e["date"] not in outlier_dates
    )
    if not obs_dates:
        return []

    # Need at least 1 prior observation, so earliest current_date > first observation
    earliest = obs_dates[0] + timedelta(days=1)
    latest = completion_dt - timedelta(days=8)  # avoid overlap with 1wk fixed lead

    if earliest >= latest:
        return []

    total_days = (latest - earliest).days
    if total_days < 1:
        return []

    # Sample n random days in the range
    random_offsets = np.random.choice(total_days, size=min(n, total_days), replace=False)
    random_offsets.sort()

    results = []
    for offset in random_offsets:
        current_dt = earliest + timedelta(days=int(offset))
        lead_days = (completion_dt - current_dt).days

        seq = gather_weekly_sequence(weekly_list, current_dt, outlier_dates)
        if seq is None:
            continue

        results.append({
            **seq,
            "current_date": current_dt.strftime("%Y-%m-%d"),
            "lead_time_days": lead_days,
            "is_fixed_lead": False,
            "lead_time_bucket": 4,  # random bucket
        })
    return results


def sample_negative_completions(weekly_list, outlier_dates, n=1):
    """Sample n fake completion dates for negatives from the later portion of timeline.

    Picks dates from the last 60% of observations to ensure enough prior history
    for input sequences at all fixed lead times.
    """
    dates = sorted(
        parse_date(e["date"]) for e in weekly_list
        if parse_date(e.get("date")) and e["date"] not in outlier_dates
    )
    if len(dates) < 5:
        return []

    # Sample from later portion — need enough history before the fake completion
    start_idx = max(3, int(len(dates) * 0.4))
    candidates = dates[start_idx:]
    if not candidates:
        return []
    n = min(n, len(candidates))
    indices = np.random.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(indices)]


def in_geo_holdout(lat, lon, geo_boxes):
    for box in geo_boxes:
        if (box["lat_min"] <= lat <= box["lat_max"] and
                box["lon_min"] <= lon <= box["lon_max"]):
            return True
    return False


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="Prepare V5 completion-aligned dataset")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "train.pt"))
    parser.add_argument("--min-text-sim", type=float, default=None,
                        help="Filter positive samples below this text_cos_sim threshold")
    parser.add_argument("--n-random", type=int, default=N_RANDOM_LEAD,
                        help="Number of random lead-time samples per location")
    parser.add_argument("--sensor", default="ps", choices=["ps", "s2", "remoteclip", "remoteclip_s2"],
                        help="Sensor/model: ps (PS RS-SigLIP), s2 (S2 RS-SigLIP), remoteclip (PS RemoteCLIP FT), remoteclip_s2 (S2 RemoteCLIP FT)")
    args = parser.parse_args()

    # Widen completion search window for S2 (monthly composites, ~30 day spacing)
    global COMPLETION_SEARCH_DAYS, EMB_DIM
    if args.sensor in ("s2", "remoteclip_s2"):
        COMPLETION_SEARCH_DAYS = 21
    if args.sensor in ("remoteclip", "remoteclip_s2"):
        EMB_DIM = 768

    print("=" * 70)
    print(f"PREPARING DATASET (completion-aligned GT, sensor={args.sensor})")
    print("=" * 70)

    # ── Phase 1: Load all data sources ──
    print("\nPhase 1: Loading data...")

    with open(os.path.join("configs", "splits.json")) as f:
        splits_config = json.load(f)
    oov_pos = set(splits_config["oov_positive_classes"])
    oov_neg_base = set(splits_config["oov_negative_base_classes"])
    oov_neg_all = set()
    for b in oov_neg_base:
        oov_neg_all.add(b)
        for i in range(2, 4):
            oov_neg_all.add(f"{b} {i}")
    geo_boxes = splits_config["geo_holdout_boxes"]

    verified_df = pd.read_csv(os.path.join(DATA_DIR, "labels_verified.csv"))
    verified_s2ids = set(str(s) for s in verified_df["s2_id"].values)

    pos_df = pd.read_csv(os.path.join(DATA_DIR, "labels_positive.csv"))
    neg_df = pd.read_csv(os.path.join(DATA_DIR, "labels_negative.csv"))

    sensor_prefix = {"ps": "ps_rssiglip", "s2": "s2_rssiglip", "remoteclip": "ps_remoteclipft", "remoteclip_s2": "s2_remoteclipft"}[args.sensor]
    pos_weekly_path = os.path.join(CONSOLIDATED_DIR, f"{sensor_prefix}_positive_weekly.json")
    neg_weekly_path = os.path.join(CONSOLIDATED_DIR, f"{sensor_prefix}_negative_weekly.json")
    print(f"  Sensor: {args.sensor} ({pos_weekly_path})")

    with open(pos_weekly_path) as f:
        pos_weekly = json.load(f)
    with open(neg_weekly_path) as f:
        neg_weekly = json.load(f)
    # Use hybrid completion dates if available, fallback to v4
    hybrid_path = os.path.join(DATA_DIR, "completion_dates.json")
    v4_path = os.path.join(DATA_DIR, "completion_dates_v4.json")
    if os.path.exists(hybrid_path):
        with open(hybrid_path) as f:
            completion_dates = json.load(f)
        print(f"  Using hybrid completion dates ({len(completion_dates)} locs)")
    else:
        with open(v4_path) as f:
            completion_dates = json.load(f)
        print(f"  Using V4 completion dates ({len(completion_dates)} locs)")
    with open(os.path.join(DATA_DIR, "before_after_rssiglip.json")) as f:
        ba_embs = json.load(f).get("embeddings", {})
    text_emb_file = "text_remoteclipft.json" if args.sensor in ("remoteclip", "remoteclip_s2") else "text_rssiglip.json"
    with open(os.path.join(DATA_DIR, text_emb_file)) as f:
        text_data = json.load(f)

    # Build text embedding lookup: class_name -> np.array
    text_class_embs = {}
    for cls_name, emb in zip(text_data["classes"], text_data["embeddings"]):
        text_class_embs[cls_name] = np.array(emb, dtype=np.float32)

    # Build location metadata
    loc_meta = {}
    for _, row in pd.concat([pos_df, neg_df]).iterrows():
        lid = f"{row['event_name'].replace(' ', '_')}_{row['s2_id']}"
        loc_meta[lid] = {
            "class": row["event_name"],
            "s2_id": str(row["s2_id"]),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "is_negative": row["event_category"] == "negative",
        }

    print(f"  Positive weekly: {len(pos_weekly)} locs")
    print(f"  Negative weekly: {len(neg_weekly)} locs")
    print(f"  Completion dates: {len(completion_dates)} locs")
    print(f"  Before/after embeddings: {len(ba_embs)} locs")
    print(f"  Text classes: {len(text_class_embs)} classes")
    print(f"  Verified s2_ids: {len(verified_s2ids)}")

    # ── Phase 2: Build positive samples ──
    print("\nPhase 2: Building positive samples...")
    samples = []
    skipped = Counter()
    text_sims_all = []
    lead_time_counts = Counter()
    np.random.seed(42)

    for loc_id, w_list in pos_weekly.items():
        meta = loc_meta.get(loc_id)
        if not meta:
            skipped["no_meta"] += 1
            continue

        cls, s2_id = meta["class"], meta["s2_id"]
        lat, lon = meta["lat"], meta["lon"]
        is_oov = cls in oov_pos
        is_verified = s2_id in verified_s2ids
        is_geo = in_geo_holdout(lat, lon, geo_boxes)

        if is_oov and not is_verified:
            skipped["oov_excluded"] += 1
            continue
        if is_geo and not is_verified and not is_oov:
            skipped["geo_excluded"] += 1
            continue

        split = "oov" if is_oov else ("test" if is_verified else "train")

        # Get completion date: max(M2, hybrid)
        hybrid_date_str = completion_dates.get(loc_id, {}).get("completion_date")
        m2_date_str = None
        ba = ba_embs.get(loc_id)
        emb_dim_check = len(w_list[0]["embedding"]) if w_list else 0
        ba_dim_check = len(ba["before"]) if ba and ba.get("before") else 0
        dims_match = (ba and ba.get("before") and ba.get("after")
                      and emb_dim_check > 0 and emb_dim_check == ba_dim_check
                      and len(ba["after"]) == emb_dim_check)
        if dims_match:
            before_arr = np.array(ba["before"], dtype=np.float32)
            after_arr = np.array(ba["after"], dtype=np.float32)
            m2_outliers = td_detect_outliers(w_list)
            m2_series = build_m2_series(w_list, before_arr, after_arr, m2_outliers)
            if len(m2_series) >= 5:
                m2_result = detect_permanent(m2_series)
                if m2_result:
                    m2_date_str = m2_result["completion_date"]

        if m2_date_str and hybrid_date_str:
            comp_date_str = max(m2_date_str, hybrid_date_str)
        elif hybrid_date_str:
            comp_date_str = hybrid_date_str
        elif m2_date_str:
            comp_date_str = m2_date_str
        else:
            skipped["no_completion_date"] += 1
            continue

        comp_dt = parse_date(comp_date_str)
        if comp_dt is None:
            skipped["no_completion_date"] += 1
            continue

        # Detect outlier dates
        outlier_dates = detect_outlier_dates(w_list)

        # === TRY-THIS HOOKS (activated by try_this_prepare_datasets.py) ===
        # 1) Per-category text-prompt M2 override (before the +2 shift)
        _pcfg = globals().get("_TRYTHIS_PROMPT_CFG")
        _pembs = globals().get("_TRYTHIS_PROMPT_EMBS")
        _c2c = globals().get("_TRYTHIS_CLASS_TO_CAT")
        if _pcfg and _pembs and _c2c and cls in _c2c:
            pc = _pcfg.get(_c2c[cls])
            if pc and pc["before"] in _pembs and pc["after"] in _pembs:
                _mo = td_detect_outliers(w_list)
                _series = build_m2_series(w_list,
                                          _pembs[pc["before"]],
                                          _pembs[pc["after"]],
                                          _mo)
                if _series and len(_series) >= 4:
                    from scripts.completion_detection import smooth_median as _sm
                    _vals = _sm([r[2] for r in _series])
                    _k = 4; _margin = 0.03; _off = 1
                    for _i in range(len(_vals) - _k + 1):
                        if all(v > _margin for v in _vals[_i:_i+_k]):
                            comp_dt = _series[min(_i + _off, len(_series) - 1)][1]
                            break
        # 2) +2 observations shift (after any override)
        if globals().get("_TRYTHIS_SHIFT_PLUS2", False):
            _obs = sorted([e for e in w_list
                           if parse_date(e.get("date")) and e["date"] not in outlier_dates],
                          key=lambda e: e["date"])
            if _obs:
                _dates = [parse_date(e["date"]) for e in _obs]
                _nearest = min(range(len(_dates)), key=lambda i: abs((_dates[i] - comp_dt).days))
                _shifted = min(_nearest + 2, len(_dates) - 1)
                comp_dt = _dates[_shifted]
        # === end TRY-THIS HOOKS ===

        # Get text embedding for this class (if available)
        text_emb = text_class_embs.get(cls)

        # Find completion embedding
        comp_emb, comp_matched_date, text_sim = find_completion_embedding(
            w_list, comp_dt, outlier_dates, text_emb=text_emb
        )
        if comp_emb is None:
            skipped["no_completion_emb"] += 1
            continue

        # Apply min text sim filter if requested
        if args.min_text_sim is not None and text_emb is not None:
            if text_sim < args.min_text_sim:
                skipped["below_text_sim"] += 1
                continue

        if text_sim > -1.0:
            text_sims_all.append(text_sim)

        # Before embedding (skip ba_embs for non-SigLIP sensors — dims won't match)
        ba = ba_embs.get(loc_id)
        if ba and ba.get("before") and len(ba["before"]) == EMB_DIM:
            before_emb = np.array(ba["before"], dtype=np.float32)
        elif w_list:
            before_emb = np.array(
                sorted(w_list, key=lambda e: e["date"])[0]["embedding"],
                dtype=np.float32,
            )
        else:
            skipped["no_before"] += 1
            continue

        # GT = weekly clean observation near completion (not Spanner composite)
        gt_completion = comp_emb
        gt_change = before_emb - gt_completion

        # Generate fixed lead-time samples
        fixed_samples = generate_fixed_lead_samples(w_list, comp_dt, outlier_dates)
        for s in fixed_samples:
            lead_time_counts[s["lead_time_bucket"]] += 1

        # Generate random lead-time samples
        random_samples = generate_random_lead_samples(
            w_list, comp_dt, outlier_dates, n=args.n_random
        )
        for s in random_samples:
            lead_time_counts[4] += 1

        all_loc_samples = fixed_samples + random_samples
        if not all_loc_samples:
            skipped["no_samples"] += 1
            continue

        for s in all_loc_samples:
            samples.append({
                **s,
                "before_emb": before_emb,
                "gt_completion": gt_completion,
                "gt_change": gt_change,
                "text_cos_sim": text_sim,
                "event_class": cls,
                "location_id": loc_id,
                "is_negative": False,
                "latitude": lat,
                "longitude": lon,
                "split": split,
                "in_geo_holdout": is_geo,
                "completion_date": comp_matched_date,
            })

    n_pos = len(samples)
    print(f"  Positive samples: {n_pos}")
    print(f"  Positive locations used: {len(set(s['location_id'] for s in samples))}")
    print(f"  Skipped: {dict(skipped)}")

    # ── Phase 3: Build negative samples ──
    print("\nPhase 3: Building negative samples...")
    neg_skipped = Counter()

    for loc_id, w_list in neg_weekly.items():
        meta = loc_meta.get(loc_id)
        if not meta:
            neg_skipped["no_meta"] += 1
            continue

        cls = meta["class"]
        lat, lon = meta["lat"], meta["lon"]
        is_oov = cls in oov_neg_all
        is_geo = in_geo_holdout(lat, lon, geo_boxes)

        split = "oov" if is_oov else ("test" if is_geo else "train")

        # Detect outlier dates
        outlier_dates = detect_outlier_dates(w_list)

        # Before embedding = earliest observation
        sorted_wl = sorted(w_list, key=lambda e: e["date"])
        before_emb = np.array(sorted_wl[0]["embedding"], dtype=np.float32)

        # Sample fake completion dates and generate lead-time samples
        fake_completions = sample_negative_completions(w_list, outlier_dates)
        if not fake_completions:
            neg_skipped["no_fake_completions"] += 1
            continue

        loc_neg_samples = []
        for comp_dt in fake_completions:
            fixed_samples = generate_fixed_lead_samples(w_list, comp_dt, outlier_dates)
            random_samples = generate_random_lead_samples(
                w_list, comp_dt, outlier_dates, n=args.n_random
            )
            loc_neg_samples.extend(fixed_samples + random_samples)

        if not loc_neg_samples:
            neg_skipped["no_samples"] += 1
            continue

        # GT: no change — completion = before
        gt_completion = before_emb.copy()
        gt_change = np.zeros(EMB_DIM, dtype=np.float32)

        for s in loc_neg_samples:
            samples.append({
                **s,
                "before_emb": before_emb,
                "gt_completion": gt_completion,
                "gt_change": gt_change,
                "text_cos_sim": -1.0,
                "lead_time_days": -1,  # override — negatives have no real lead time
                "is_fixed_lead": False,
                "lead_time_bucket": -1,
                "event_class": cls,
                "location_id": loc_id,
                "is_negative": True,
                "latitude": lat,
                "longitude": lon,
                "split": split,
                "in_geo_holdout": is_geo,
                "completion_date": None,
            })

    n_neg = len(samples) - n_pos
    print(f"  Negative samples: {n_neg}")
    print(f"  Skipped: {dict(neg_skipped)}")

    # ── Phase 4: Apply splits, balance test negatives ──
    print("\nPhase 4: Applying splits...")
    test_pos = sum(1 for s in samples if s["split"] == "test" and not s["is_negative"])
    train_neg_idx = [i for i, s in enumerate(samples) if s["split"] == "train" and s["is_negative"]]
    np.random.shuffle(train_neg_idx)
    moved = 0
    for idx in train_neg_idx[:test_pos]:
        samples[idx]["split"] = "test"
        moved += 1
    print(f"  Moved {moved} negatives to test for balance")

    # ── Phase 5: Report statistics ──
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total = len(samples)
    print(f"\nTotal: {total} samples ({n_pos} pos + {n_neg} neg)")

    # Split breakdown
    for split in ["train", "test", "oov"]:
        sl = [s for s in samples if s["split"] == split]
        pos = sum(1 for s in sl if not s["is_negative"])
        neg = sum(1 for s in sl if s["is_negative"])
        geo = sum(1 for s in sl if s["in_geo_holdout"])
        locs = len(set(s["location_id"] for s in sl))
        tok_avg = np.mean([s["n_tokens"] for s in sl]) if sl else 0
        print(f"  {split:5s}: {len(sl):6d} ({pos:5d} pos + {neg:5d} neg) | "
              f"{locs:5d} locs | {geo:3d} geo | avg {tok_avg:.1f} tokens")

    # Lead-time distribution (positive only)
    print("\nLead-time distribution (positive samples):")
    bucket_names = {0: "1mo", 1: "3mo", 2: "6mo", 3: "9mo", 4: "random"}
    for bucket_idx in sorted(lead_time_counts.keys()):
        name = bucket_names.get(bucket_idx, f"bucket_{bucket_idx}")
        count = lead_time_counts[bucket_idx]
        pct = 100.0 * count / n_pos if n_pos > 0 else 0
        print(f"  {name:8s}: {count:6d} ({pct:5.1f}%)")

    # GT quality (text_cos_sim)
    if text_sims_all:
        sims = np.array(text_sims_all)
        print(f"\nGT quality (text_cos_sim, positive locations with text embeddings):")
        print(f"  n={len(sims)}, mean={sims.mean():.4f}, median={np.median(sims):.4f}")
        print(f"  std={sims.std():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")
        for t in [0.0, 0.05, 0.10, 0.15, 0.20]:
            pct = 100.0 * (sims >= t).sum() / len(sims)
            print(f"  >= {t:.2f}: {pct:5.1f}%")

    # Lead time stats for positives
    pos_lead_times = [s["lead_time_days"] for s in samples if not s["is_negative"]]
    if pos_lead_times:
        lt = np.array(pos_lead_times)
        print(f"\nLead-time days (positive samples):")
        print(f"  mean={lt.mean():.1f}, median={np.median(lt):.1f}, "
              f"min={lt.min()}, max={lt.max()}")

    # ── Phase 6: Assemble tensors ──
    print(f"\nPhase 6: Assembling tensors...")
    n = len(samples)

    dataset = {
        # Input
        "sequences": torch.from_numpy(np.stack([s["sequence"] for s in samples])),
        "masks": torch.from_numpy(np.stack([s["mask"] for s in samples])),
        "temporal_offsets": torch.from_numpy(np.stack([s["temporal_offsets"] for s in samples])),
        "before_embs": torch.from_numpy(np.stack([s["before_emb"] for s in samples])),

        # GT (single completion target)
        "gt_completion": torch.from_numpy(np.stack([s["gt_completion"] for s in samples])),
        "gt_change": torch.from_numpy(np.stack([s["gt_change"] for s in samples])),

        # Lead-time metadata
        "lead_time_days": torch.tensor(
            [s["lead_time_days"] for s in samples], dtype=torch.float32
        ),
        "text_cos_sim": torch.tensor(
            [s["text_cos_sim"] for s in samples], dtype=torch.float32
        ),
        "is_fixed_lead": torch.tensor(
            [s["is_fixed_lead"] for s in samples], dtype=torch.bool
        ),
        "lead_time_bucket": torch.tensor(
            [s["lead_time_bucket"] for s in samples], dtype=torch.long
        ),

        # Metadata
        "is_negative": torch.tensor([s["is_negative"] for s in samples], dtype=torch.bool),
        "in_geo_holdout": torch.tensor([s["in_geo_holdout"] for s in samples], dtype=torch.bool),
        "latitudes": torch.tensor([s["latitude"] for s in samples], dtype=torch.float32),
        "longitudes": torch.tensor([s["longitude"] for s in samples], dtype=torch.float32),
        "split_labels": [s["split"] for s in samples],
        "class_labels": [s["event_class"] for s in samples],
        "location_ids": [s["location_id"] for s in samples],
        "current_dates": [s["current_date"] for s in samples],
        "completion_dates": [s["completion_date"] for s in samples],

        # Constants
        "emb_dim": EMB_DIM,
        "max_seq_len": MAX_WEEKLY,
        "lead_time_buckets": ["1mo", "3mo", "6mo", "9mo", "random"],
    }

    torch.save(dataset, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
