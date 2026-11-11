#!/usr/bin/env python3
"""V5 Evaluation: end-to-end retrieval mAP and change detection.

For each text class query, rank ALL samples in the pool.
- Gated in (detected as changed): score = cosine_sim(emb, text_emb)
- Gated out (no change detected): score = -inf — cannot be retrieved

Positives perceived as no change are lost. Negatives perceived as change
become false positives that dilute retrieval.

4 settings, each with its own pool:
  - Seen:     non-geo test positives + non-geo negatives
  - OOV:      non-geo oov positives  + non-geo negatives
  - Geo Seen: geo test positives     + geo negatives
  - Geo OOV:  geo oov positives      + geo negatives

Usage:
    python -m scripts.evaluate --results-dir results
    python -m scripts.evaluate --results-dir results --experiment B_learn_after
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from anticipation.models.autoregressive_v4 import TemporalConditionedAR, MultiHorizonAR


LEAD_RANGES = [
    ("<1wk",   0,   7),
    ("1w-1mo", 7,  30),
    ("1-3mo", 30,  90),
    ("3-6mo", 90, 180),
    ("6-9mo",180, 270),
    ("9mo+", 270, 9999),
]

SETTINGS = [
    ("Seen",     "test", False),
    ("OOV",      "oov",  False),
    ("Geo Seen", "test", True),
    ("Geo OOV",  "oov",  True),
]

METHOD_ORDER = [
    "Oracle GT", "Static (no gate)", "Model",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_text_embeddings(path):
    with open(path) as f:
        d = json.load(f)
    embs = torch.tensor(d["embeddings"], dtype=torch.float32)
    if len(d["classes"]) != 72:
        print(f"\n  !!!!! WARNING: Expected 72 text classes but got {len(d['classes'])}! !!!!!")
        print(f"  !!!!! Check: {path} !!!!!\n")
    return d["classes"], F.normalize(embs, dim=-1)


def load_class_groups(path="configs/class_groups.json"):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    mapping = {}
    for members in d["groups"].values():
        s = set(members)
        for m in members:
            mapping[m] = s
    return mapping


# ---------------------------------------------------------------------------
# Gate (change detection)
# ---------------------------------------------------------------------------

def compute_gate_threshold(change_mags, is_neg_arr):
    """F1-optimal threshold for positive/negative separation."""
    from sklearn.metrics import f1_score, roc_auc_score

    mags = change_mags.numpy()
    labels = (~is_neg_arr).astype(int)

    if len(set(labels)) < 2:
        return 0.0, {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    auc = roc_auc_score(labels, mags)
    thresholds = np.percentile(mags, np.linspace(0, 100, 1000))
    best_f1, best_t = 0.0, 0.0
    for t in thresholds:
        f1 = f1_score(labels, (mags >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    detected = mags >= best_t
    tp = int((detected & (labels == 1)).sum())
    fp = int((detected & (labels == 0)).sum())
    fn = int((~detected & (labels == 1)).sum())
    tn = int((~detected & (labels == 0)).sum())

    return float(best_t), {
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": best_f1,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Retrieval: mAP@k with -inf gating
# ---------------------------------------------------------------------------

def compute_map_at_k(embs, gate_mask, labels, is_neg,
                     text_names, text_embs, k_values,
                     relevant_mask=None, class_to_group=None,
                     target_classes=None):
    """Rank all samples per text query. Gated-out get -inf. Return mean AP@k."""
    sims = F.normalize(embs, dim=-1) @ text_embs.T
    sims_np = sims.numpy().copy()
    sims_np[~gate_mask] = -np.inf

    if target_classes is None:
        target_classes = sorted(set(l for l, n in zip(labels, is_neg) if not n))
    target_classes = [c for c in target_classes if c in text_names]

    per_class = {}
    for cls in target_classes:
        idx = text_names.index(cls)
        scores = sims_np[:, idx]

        if class_to_group and cls in class_to_group:
            base = np.array([1 if (l in class_to_group[cls] and not n) else 0
                             for l, n in zip(labels, is_neg)])
        else:
            base = np.array([1 if (l == cls and not n) else 0
                             for l, n in zip(labels, is_neg)])

        rel = base * relevant_mask.astype(int) if relevant_mask is not None else base
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue

        order = np.argsort(-scores)
        sorted_rel = rel[order]

        aps = {}
        for k in k_values:
            rk = sorted_rel[:k]
            if rk.sum() == 0:
                aps[k] = 0.0
            else:
                prec = np.cumsum(rk) / np.arange(1, k + 1)
                aps[k] = float((prec * rk).sum() / min(n_rel, k))
        per_class[cls] = aps

    mean = {}
    for k in k_values:
        vals = [v[k] for v in per_class.values()]
        mean[k] = float(np.mean(vals)) if vals else 0.0

    return per_class, mean


def compute_recall_at_k(embs, gate_mask, labels, is_neg,
                        text_names, text_embs, k_values,
                        relevant_mask=None, class_to_group=None,
                        target_classes=None):
    """Mean Recall@k: for each class, is there ≥1 correct hit in top-k?"""
    sims = F.normalize(embs, dim=-1) @ text_embs.T
    sims_np = sims.numpy().copy()
    sims_np[~gate_mask] = -np.inf

    if target_classes is None:
        target_classes = sorted(set(l for l, n in zip(labels, is_neg) if not n))
    target_classes = [c for c in target_classes if c in text_names]

    per_class = {}
    for cls in target_classes:
        idx = text_names.index(cls)
        scores = sims_np[:, idx]

        if class_to_group and cls in class_to_group:
            base = np.array([1 if (l in class_to_group[cls] and not n) else 0
                             for l, n in zip(labels, is_neg)])
        else:
            base = np.array([1 if (l == cls and not n) else 0
                             for l, n in zip(labels, is_neg)])

        rel = base * relevant_mask.astype(int) if relevant_mask is not None else base
        if rel.sum() == 0:
            continue

        order = np.argsort(-scores)
        sorted_rel = rel[order]

        recalls = {}
        for k in k_values:
            recalls[k] = float(sorted_rel[:k].sum() > 0)
        per_class[cls] = recalls

    mean = {}
    for k in k_values:
        vals = [v[k] for v in per_class.values()]
        mean[k] = float(np.mean(vals)) if vals else 0.0

    return per_class, mean


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------

class _EvalDataset(Dataset):
    def __init__(self, data, indices):
        self.data, self.indices = data, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        query_offset = self.data["lead_time_days"][i].clone()
        if self.data["is_negative"][i]:
            query_offset = torch.tensor(30.0)
        return {
            "sequence": self.data["sequences"][i],
            "mask": self.data["masks"][i],
            "temporal_offsets": self.data["temporal_offsets"][i],
            "before_emb": self.data["before_embs"][i],
            "latitude": self.data["latitudes"][i],
            "longitude": self.data["longitudes"][i],
            "query_offset": query_offset,
            "lead_time_bucket": self.data["lead_time_bucket"][i],
        }


@torch.no_grad()
def run_inference(model, data, indices, device, arch="B", batch_size=64):
    loader = DataLoader(_EvalDataset(data, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_preds = []
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

        if arch == "A":
            preds = model(b["sequence"], b["mask"], b["before_emb"],
                          temporal_offsets=b["temporal_offsets"],
                          latitude=b["latitude"], longitude=b["longitude"])
            B, H, D = preds.shape
            head_idx = b["lead_time_bucket"].long().clone()
            head_idx[head_idx < 0] = 1
            head_idx[head_idx >= H] = 1
            gather_idx = head_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)
            pred = preds.gather(1, gather_idx).squeeze(1)
        else:
            pred = model(b["sequence"], b["mask"], b["before_emb"],
                         temporal_offsets=b["temporal_offsets"],
                         query_offset=b["query_offset"],
                         latitude=b["latitude"], longitude=b["longitude"])

        all_preds.append(pred.cpu())
    return torch.cat(all_preds)


def load_model(exp_name, results_dir, emb_dim, max_seq_len, device):
    from scripts.train import EXPERIMENTS
    exp = EXPERIMENTS[exp_name]

    ckpt = os.path.join(results_dir, exp_name, "best_model.pt")
    if not os.path.exists(ckpt):
        return None, exp

    cfg_path = os.path.join(results_dir, exp_name, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    nl = cfg.get("n_layers", 4)
    nh = cfg.get("n_heads", 8)
    ff = cfg.get("ff_dim", 2048)
    do = cfg.get("dropout", 0.2)

    arch = exp.get("arch", "B")
    sinusoidal = exp.get("sinusoidal", True)

    if arch == "A":
        model = MultiHorizonAR(
            emb_dim=emb_dim, n_heads=nh, n_layers=nl, ff_dim=ff,
            dropout=do, max_seq_len=max_seq_len,
            use_sinusoidal_temporal=sinusoidal,
            use_diffs=exp.get("diffs", False),
        )
    elif arch == "JEPA":
        model = JEPAConditionedAR(
            emb_dim=emb_dim, n_heads=nh,
            n_encoder_layers=nl, n_predictor_layers=nl,
            ff_dim=ff, dropout=do, max_seq_len=max_seq_len,
            use_diffs=exp.get("diffs", False),
            use_learned_input_pos=not sinusoidal,
        )
    else:
        model = TemporalConditionedAR(
            emb_dim=emb_dim, n_heads=nh, n_layers=nl, ff_dim=ff,
            dropout=do, max_seq_len=max_seq_len,
            use_diffs=exp.get("diffs", False),
            use_learned_input_pos=not sinusoidal,
        )

    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device), strict=False)
    model.to(device).eval()
    return model, exp


# ---------------------------------------------------------------------------
# External model loading (V7 etc from eval_models/)
# ---------------------------------------------------------------------------

# Models in eval_models/ that map to standard V5 experiment configs
EXTERNAL_MODELS = {
    "v7_B_after_change_combo": {"arch": "B", "gt": "gt_completion", "diffs": False, "sinusoidal": False},
    "v7_B_after_simple":       {"arch": "B_simple", "gt": "gt_completion", "diffs": False, "sinusoidal": False},
    "v7_B_after":              {"arch": "B", "gt": "gt_completion", "diffs": False, "sinusoidal": True},
    "v7_B_after_vel":          {"arch": "B", "gt": "gt_completion", "diffs": True, "sinusoidal": True},
    "v7_B_after_multi":        {"arch": "B", "gt": "gt_completion", "diffs": False, "sinusoidal": True},
    "v7_B_after_old_vel":      {"arch": "B", "gt": "gt_completion", "diffs": True, "sinusoidal": True},
    "v5_B_learn_after":        {"arch": "B", "gt": "gt_completion", "diffs": False, "sinusoidal": False},
    "v6_B_after":              {"arch": "B", "gt": "gt_completion", "diffs": False, "sinusoidal": True},
    "v6_B_after_diffs":        {"arch": "B", "gt": "gt_completion", "diffs": True, "sinusoidal": True},
}


def load_external_model(name, eval_models_dir, emb_dim, max_seq_len, device):
    """Load a model from eval_models/ using EXTERNAL_MODELS config."""
    exp = EXTERNAL_MODELS[name]
    ckpt = os.path.join(eval_models_dir, name, "best_model.pt")
    if not os.path.exists(ckpt):
        return None, exp

    cfg_path = os.path.join(eval_models_dir, name, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    nl = cfg.get("n_layers", 4)
    nh = cfg.get("n_heads", 8)
    ff = cfg.get("ff_dim", 2048)
    do = cfg.get("dropout", 0.2)

    arch = exp.get("arch", "B")

    if arch == "B_simple":
        model = SimpleTemporalConditionedAR(
            emb_dim=emb_dim, n_heads=nh, n_layers=nl, ff_dim=ff,
            dropout=do, max_seq_len=max_seq_len,
            use_diffs=exp.get("diffs", False),
            use_learned_input_pos=not exp.get("sinusoidal", True),
        )
    else:
        model = TemporalConditionedAR(
            emb_dim=emb_dim, n_heads=nh, n_layers=nl, ff_dim=ff,
            dropout=do, max_seq_len=max_seq_len,
            use_diffs=exp.get("diffs", False),
            use_learned_input_pos=not exp.get("sinusoidal", True),
        )

    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device), strict=False)
    model.to(device).eval()
    return model, exp


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_experiment(exp_name, model, exp_config, data, eval_indices,
                    text_names, text_embs, class_to_group, device):
    """Run inference, then evaluate all methods on all 4 settings."""
    gt_mode = exp_config["gt"]
    k_values = [5, 10, 20, 30]

    N = len(eval_indices)
    class_labels = [data["class_labels"][i] for i in eval_indices]
    is_neg = np.array([bool(data["is_negative"][i]) for i in eval_indices])
    split_labels = np.array([data["split_labels"][i] for i in eval_indices])
    in_geo = data["in_geo_holdout"][eval_indices].numpy().astype(bool)
    before_embs = data["before_embs"][eval_indices]
    gt_completion = data["gt_completion"][eval_indices]
    lead_days = data["lead_time_days"][eval_indices].numpy()
    pos_mask = ~is_neg

    # Last observation (current date) embedding = last valid token per sample
    seqs = data["sequences"][eval_indices]
    masks = data["masks"][eval_indices]
    last_obs = torch.zeros_like(before_embs)
    for i in range(N):
        n_valid = int(masks[i].sum().item())
        if n_valid > 0:
            last_obs[i] = seqs[i, n_valid - 1]
        else:
            last_obs[i] = before_embs[i]

    # 1. Run inference on everything
    arch = exp_config.get("arch", "B")
    run_arch = "B" if arch == "B_simple" else arch
    print(f"  Running inference (Arch {arch})...")
    raw_preds = run_inference(model, data, eval_indices, device, arch=run_arch)

    # 2. Compute pred_after and change magnitudes
    dir_only = exp_config.get("direction_only", False)
    if gt_mode == "gt_change":
        change_mags = raw_preds.norm(dim=-1)
        if dir_only:
            pred_after = before_embs - F.normalize(raw_preds, dim=-1)
        else:
            pred_after = before_embs - raw_preds
    else:
        pred_after = raw_preds
        change_mags = (before_embs - raw_preds).norm(dim=-1)

    # 3. Gate threshold (on full eval set) — override with --eps if provided
    _eps_override = globals().get("_EPS_OVERRIDE")
    if _eps_override is not None:
        from sklearn.metrics import f1_score, precision_score, recall_score
        mags_np = change_mags.numpy()
        labels_np = (~is_neg).numpy().astype(int)
        detected = mags_np >= _eps_override
        threshold = float(_eps_override)
        det_stats = {
            "accuracy": float((detected == labels_np).mean()),
            "precision": float(precision_score(labels_np, detected, zero_division=0)),
            "recall": float(recall_score(labels_np, detected, zero_division=0)),
            "f1": float(f1_score(labels_np, detected, zero_division=0)),
        }
        print(f"    Using fixed gate ε = {threshold:.4f}  (F1 = {det_stats['f1']:.3f})")
    else:
        threshold, det_stats = compute_gate_threshold(change_mags, is_neg)
        print(f"    Auto-swept F1-optimal ε = {threshold:.4f}  (F1 = {det_stats['f1']:.3f})")
    gate_pass = (change_mags >= threshold).numpy()

    # 4. Build oracle embeddings (GT for pos, before for neg)
    oracle_embs = before_embs.clone()
    oracle_embs[pos_mask] = gt_completion[pos_mask]

    # 5. Evaluate each setting
    results_by_setting = {}
    for sname, split, is_geo in SETTINGS:
        # Pool: all samples in this region (geo or non-geo)
        pool_mask = in_geo if is_geo else ~in_geo
        idx = np.where(pool_mask)[0]
        if len(idx) == 0:
            continue

        p_before = before_embs[idx]
        p_oracle = oracle_embs[idx]
        p_pred   = pred_after[idx]
        p_neg    = is_neg[idx]
        p_pos    = ~p_neg
        p_gate   = gate_pass[idx]
        p_labels = [class_labels[j] for j in idx]
        p_splits = split_labels[idx]
        p_all    = np.ones(len(idx), dtype=bool)

        # Relevant positives: those in the specified split
        relevant = p_pos & (p_splits == split)
        n_rel = int(relevant.sum())
        if n_rel == 0:
            continue

        tc = sorted(set(p_labels[j] for j in range(len(idx)) if relevant[j]))

        p_last = last_obs[idx]
        methods = {
            "Oracle GT":          (p_oracle,       p_pos.copy()),
            "Static (no gate)":   (p_last,         p_all.copy()),
            "Model":              (p_pred,         p_gate),
        }

        # Detection metrics for this setting's relevant positives
        detected = p_gate
        tp = int((detected & relevant).sum())
        fp = int((detected & p_neg).sum())
        fn = int((~detected & relevant).sum())
        det_prec = tp / max(tp + fp, 1)
        det_rec = tp / max(tp + fn, 1)
        det_f1 = 2 * det_prec * det_rec / max(det_prec + det_rec, 1e-8)

        setting_results = {"n_pool": len(idx), "n_pos": int(p_pos.sum()),
                           "n_neg": int(p_neg.sum()), "n_relevant": n_rel,
                           "det_f1": det_f1, "det_prec": det_prec,
                           "det_rec": det_rec}
        for method_name in METHOD_ORDER:
            embs, gate = methods[method_name]
            _, grouped = compute_map_at_k(
                embs, gate, p_labels, p_neg,
                text_names, text_embs, k_values,
                relevant_mask=relevant, class_to_group=class_to_group,
                target_classes=tc)
            setting_results[method_name] = grouped

        results_by_setting[sname] = setting_results

    # 6. Lead-time breakdown: separate mini-pool per bucket
    #    Each pool = bucket's non-geo positives + all non-geo negatives
    nongeo_neg_idx = np.where(~in_geo & is_neg)[0]
    bucket_map = {}
    for rname, lo, hi in LEAD_RANGES:
        bucket_pos_idx = np.where(~in_geo & pos_mask &
                                  (lead_days >= lo) & (lead_days < hi))[0]
        n_bucket = len(bucket_pos_idx)
        if n_bucket == 0:
            bucket_map[rname] = {"model": 0.0, "oracle": 0.0, "n_pos": 0}
            continue
        # Mini pool: this bucket's positives + all negatives
        pool_idx = np.concatenate([bucket_pos_idx, nongeo_neg_idx])
        p_oracle_b = oracle_embs[pool_idx]
        p_pred_b = pred_after[pool_idx]
        p_last_b = last_obs[pool_idx]
        p_neg_b = is_neg[pool_idx]
        p_pos_b = ~p_neg_b
        p_gate_b = gate_pass[pool_idx]
        p_all_b = np.ones(len(pool_idx), dtype=bool)
        p_labels_b = [class_labels[j] for j in pool_idx]

        _, m_mean = compute_map_at_k(
            p_pred_b, p_gate_b, p_labels_b, p_neg_b,
            text_names, text_embs, [10],
            class_to_group=class_to_group)
        _, o_mean = compute_map_at_k(
            p_oracle_b, p_pos_b.copy(), p_labels_b, p_neg_b,
            text_names, text_embs, [10],
            class_to_group=class_to_group)
        _, s_mean = compute_map_at_k(
            p_last_b, p_all_b, p_labels_b, p_neg_b,
            text_names, text_embs, [10],
            class_to_group=class_to_group)
        bucket_map[rname] = {
            "model": m_mean[10], "oracle": o_mean[10], "static": s_mean[10],
            "n_pos": n_bucket,
        }

    return {
        "pool_size": N,
        "detection": det_stats,
        "settings": results_by_setting,
        "lead_time_buckets": bucket_map,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_combined_table(title, seen_key, oov_key, settings, pool_desc):
    """Print a combined table: Detection F1 | Seen mAP | Unseen mAP."""
    k_values = [5, 10, 20, 30]
    seen = settings.get(seen_key, {})
    oov = settings.get(oov_key, {})
    if not seen and not oov:
        return

    print(f"\n  {title} ({pool_desc})")
    n_seen = seen.get("n_relevant", 0)
    n_oov = oov.get("n_relevant", 0)

    # Header
    seen_label = f"Seen ({n_seen})"
    oov_label = f"Unseen ({n_oov})"
    print(f"    {'':22s}  {seen_label:^28s}  |  {oov_label:^28s}")
    hdr = f"    {'Method':22s} "
    for _ in range(2):
        hdr += f" {'F1':>5s}"
        for k in k_values:
            hdr += f" {'mAP@'+str(k):>6s}"
        hdr += "  |"
    print(hdr.rstrip(" |"))
    print(f"    {'-'*22}  {'-'*28}  +{'-'*28}")

    # Rows
    baselines = {"Oracle GT", "Static (no gate)"}
    for method in METHOD_ORDER:
        line = f"    {method:22s} "
        for s in [seen, oov]:
            if method in baselines:
                line += f" {'—':>5s}"
            else:
                line += f" {s.get('det_f1', 0):5.3f}"
            grouped = s.get(method, {})
            for k in k_values:
                line += f" {grouped.get(k, 0):6.3f}"
            line += "  |"
        print(line.rstrip(" |"))


def print_results(exp_name, results):
    print(f"\n{'='*80}")
    print(f"  {exp_name}")
    print(f"{'='*80}")

    settings = results.get("settings", {})

    # Test Set table (non-geo pool)
    seen = settings.get("Seen", {})
    pool_desc = f"pool={seen.get('n_pool',0)}: {seen.get('n_pos',0)} pos + {seen.get('n_neg',0)} neg"
    _print_combined_table("Test Set", "Seen", "OOV", settings, pool_desc)

    # Geo Test Set table (geo pool)
    geo_seen = settings.get("Geo Seen", {})
    geo_desc = f"pool={geo_seen.get('n_pool',0)}: {geo_seen.get('n_pos',0)} pos + {geo_seen.get('n_neg',0)} neg"
    _print_combined_table("Geo Test Set", "Geo Seen", "Geo OOV", settings, geo_desc)

    # Lead-time breakdown
    buckets = results.get("lead_time_buckets", {})
    if buckets:
        rnames = [r[0] for r in LEAD_RANGES]
        print(f"\n  mAP@10 by lead-time:")
        hdr = f"    {'':8s}"
        for rn in rnames:
            hdr += f" {rn:>7s}"
        print(hdr)
        line = f"    {'Oracle':8s}"
        for rn in rnames:
            line += f" {buckets.get(rn,{}).get('oracle',0):7.3f}"
        print(line)
        line = f"    {'Static':8s}"
        for rn in rnames:
            line += f" {buckets.get(rn,{}).get('static',0):7.3f}"
        print(line)
        line = f"    {'Model':8s}"
        for rn in rnames:
            line += f" {buckets.get(rn,{}).get('model',0):7.3f}"
        print(line)
        line = f"    {'n_pos':8s}"
        for rn in rnames:
            line += f" {buckets.get(rn,{}).get('n_pos',0):7d}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V5 Evaluation")
    parser.add_argument("--dataset", default="data/test.pt")
    parser.add_argument("--results-dir", default="results_v5")
    parser.add_argument("--text-emb", default="data/text_rssiglip.json")
    parser.add_argument("--experiment", nargs="*", default=None)
    parser.add_argument("--eval-models-dir", default="eval_models",
                        help="Directory with external models (V7 etc)")
    parser.add_argument("--external-only", action="store_true",
                        help="Only evaluate external models from eval-models-dir")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results (default: results-dir)")
    parser.add_argument("--require-gpu", action="store_true",
                        help="Exit if GPU is not available (prevents CPU OOM)")
    parser.add_argument("--eps", default="auto",
                        help="Gate threshold: 'auto' sweeps F1-optimal on pool (default), "
                             "or pass a float (paper's is 0.245)")
    args = parser.parse_args()

    # Push --eps into module scope for the evaluator to consume
    if args.eps != "auto":
        globals()["_EPS_OVERRIDE"] = float(args.eps)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if args.require_gpu and device.type == "cpu":
        print("ERROR: --require-gpu set but no GPU available. Exiting to avoid CPU OOM.")
        return

    print("Loading dataset...")
    data = torch.load(args.dataset, weights_only=False)
    print(f"  {len(data['split_labels'])} samples, emb_dim={data['emb_dim']}")

    print("Loading text embeddings...")
    text_names, text_embs = load_text_embeddings(args.text_emb)
    print(f"  {len(text_names)} text classes")

    class_to_group = load_class_groups()
    print(f"  {len(class_to_group)} classes in "
          f"{len(set(frozenset(v) for v in class_to_group.values()))} groups")

    # Build eval index sets
    all_eval = [i for i, s in enumerate(data["split_labels"]) if s in ("test", "oov")]
    buckets = data["lead_time_bucket"]
    neg = data["is_negative"]
    fixed_eval = [i for i in all_eval if neg[i] or buckets[i].item() in (0, 1, 2, 3)]

    all_eval_t = torch.tensor(all_eval)
    fixed_eval_t = torch.tensor(fixed_eval)

    print(f"  Eval (all):   {len(all_eval)} samples")
    print(f"  Eval (fixed): {len(fixed_eval)} samples")
    for sname, split, geo in SETTINGS:
        count = sum(1 for i in all_eval
                    if data["split_labels"][i] == split
                    and bool(data["in_geo_holdout"][i]) == geo
                    and not neg[i])
        print(f"    {sname}: {count} positives")

    # --- Load V5 experiments (from results-dir) ---
    models = {}
    if not args.external_only:
        from scripts.train import EXPERIMENTS
        if args.experiment:
            exp_names = [e for e in args.experiment if e in EXPERIMENTS]
        else:
            exp_names = [e for e in EXPERIMENTS
                         if os.path.exists(os.path.join(args.results_dir, e, "best_model.pt"))]

        for exp_name in sorted(exp_names):
            model, exp_config = load_model(exp_name, args.results_dir,
                                           data["emb_dim"], data["max_seq_len"], device)
            if model is None:
                print(f"  Skipping {exp_name} (no best_model.pt)")
                continue
            models[exp_name] = (model, exp_config)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Loaded {exp_name}: Arch {exp_config.get('arch','?')}, {n_params/1e6:.1f}M params")

    # --- Load external models (from eval-models-dir) ---
    if args.eval_models_dir and os.path.isdir(args.eval_models_dir):
        ext_names = args.experiment if args.experiment else None
        for name, exp in EXTERNAL_MODELS.items():
            if ext_names and name not in ext_names:
                continue
            ckpt = os.path.join(args.eval_models_dir, name, "best_model.pt")
            if not os.path.exists(ckpt):
                continue
            model, exp_config = load_external_model(
                name, args.eval_models_dir,
                data["emb_dim"], data["max_seq_len"], device)
            if model is None:
                continue
            models[name] = (model, exp_config)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Loaded {name} (external): Arch {exp_config.get('arch','?')}, {n_params/1e6:.1f}M params")

    if not models:
        print("\nNo models found!")
        return

    b_exps = sorted([e for e, (_, cfg) in models.items()
                     if cfg.get("arch", "B") in ("B", "B_simple", "JEPA")])
    a_exps = sorted([e for e, (_, cfg) in models.items()
                     if cfg.get("arch") == "A"])
    print(f"\nArch A experiments: {a_exps}")
    print(f"Arch B experiments: {b_exps}")

    all_results = {}

    # --- Table 1: All experiments on fixed-horizon eval set ---
    print(f"\n{'='*80}")
    print(f"  TABLE 1: Fixed-horizon eval ({len(fixed_eval)} samples)")
    print(f"{'='*80}")
    for exp_name in sorted(models.keys()):
        model, exp_config = models[exp_name]
        results = eval_experiment(exp_name, model, exp_config, data, fixed_eval_t,
                                  text_names, text_embs, class_to_group, device)
        all_results[f"{exp_name}_fixed"] = results
        print_results(exp_name, results)

    # --- Table 2: Arch B on full eval set (includes random horizons) ---
    if b_exps:
        print(f"\n{'='*80}")
        print(f"  TABLE 2: Arch B — full eval ({len(all_eval)} samples)")
        print(f"{'='*80}")
        for exp_name in b_exps:
            if exp_name not in models:
                continue
            model, exp_config = models[exp_name]
            results = eval_experiment(exp_name, model, exp_config, data, all_eval_t,
                                      text_names, text_embs, class_to_group, device)
            all_results[f"{exp_name}_all"] = results
            print_results(exp_name, results)

    # Save
    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_v5.json")

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
