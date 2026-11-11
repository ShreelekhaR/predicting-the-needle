#!/usr/bin/env python3
"""
V9 training: proper mAP logging with correct magnitude gating.

Changes from train_v5.py:
  - mAP uses (before - pred).norm() for gating (not pred.norm())
  - Logs Seen mAP@10, OOV mAP@10 every 5 epochs
  - Saves best model by OOV mAP@10 (best_oov.pt) and by Seen mAP@10 (best_seen.pt)
  - Checkpoints every 5 epochs
  - Supports --predict-time

Usage:
  python -m scripts.train --experiment B_learn_after_diffs --dataset data/train.pt --results-dir results_v9 --epochs 200
  python -m scripts.train --experiment B_learn_after_diffs --dataset data/train.pt --results-dir results_v9 --predict-time --epochs 200
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anticipation.models.dataset_v5 import AnticipationDatasetV5
from anticipation.models.autoregressive_v4 import TemporalConditionedAR
from anticipation.models.lstm_v9 import LSTMPredictorV9
from anticipation.models.gru_v9 import GRUPredictorV9

# Guard against system-cuDNN vs bundled-cuDNN version mismatch on some workbenches
# (only affects RNN paths — transformer runs fine either way).
# Opt-in via env var; leave cuDNN on by default.
if os.environ.get("DISABLE_CUDNN"):
    torch.backends.cudnn.enabled = False

DATA_DIR = "data"
DATASET_FILE = os.path.join(DATA_DIR, "train.pt")
RESULTS_DIR = "results_v9"

EXPERIMENTS = {
    "B_learn_after":         {"gt": "gt_completion", "diffs": False, "sinusoidal": False},
    "B_learn_after_diffs":   {"gt": "gt_completion", "diffs": True,  "sinusoidal": False},
    "B_sin_after":           {"gt": "gt_completion", "diffs": False, "sinusoidal": True},
    "B_sin_after_diffs":     {"gt": "gt_completion", "diffs": True,  "sinusoidal": True},
}

LEAD_BUCKETS = ["1mo", "3mo", "6mo", "9mo", "random"]


def cosine_loss(pred, target):
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def magnitude_mse(pred, target):
    return F.mse_loss(pred.norm(dim=-1), target.norm(dim=-1))


def compute_map_at_k(preds, last_obs_embs, text_embs, text_class_names,
                     class_labels, is_negative, k=10):
    """Compute mAP@k with (last_obs - pred).norm() gating."""
    change_mags = (last_obs_embs - preds).norm(dim=-1)

    # Sweep threshold for best F1
    best_f1, best_thresh = 0, change_mags.median().item()
    for pct in range(5, 96, 5):
        t = np.percentile(change_mags.numpy(), pct)
        pred_pos = change_mags >= t
        tp = (pred_pos & ~is_negative).sum().item()
        fp = (pred_pos & is_negative).sum().item()
        fn = (~pred_pos & ~is_negative).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    # Retrieval with gating
    sims = F.cosine_similarity(preds.unsqueeze(1), text_embs.unsqueeze(0), dim=-1)
    gated_sims = sims.clone()
    gated_sims[change_mags < best_thresh] = -float('inf')

    aps = []
    for c_idx, c_name in enumerate(text_class_names):
        labels = torch.tensor([1 if cl == c_name else 0 for cl in class_labels])
        if labels.sum() == 0:
            continue
        scores = gated_sims[:, c_idx]
        ranked = labels[scores.argsort(descending=True)][:k]
        prec_at = [(ranked[:i+1].sum().float()/(i+1)).item()
                   for i in range(len(ranked)) if ranked[i]]
        aps.append(np.mean(prec_at) if prec_at else 0.0)

    return np.mean(aps) if aps else 0.0, best_f1


@torch.no_grad()
def eval_and_map(model, dataset, device, text_embs, text_class_names, batch_size=64):
    """Run inference + compute mAP@10 with correct gating."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_cos = []

    for batch in loader:
        seq = batch["sequence"].to(device)
        mask = batch["mask"].to(device)
        before = batch["before_emb"].to(device)
        offsets = batch["temporal_offsets"].to(device)
        lat = batch["latitude"].to(device)
        lon = batch["longitude"].to(device)
        target = batch["target"].to(device)
        query = batch["query_offset"].to(device)

        pred = model(seq, mask, before, temporal_offsets=offsets,
                     query_offset=query, latitude=lat, longitude=lon)

        cos = F.cosine_similarity(pred, target, dim=-1)
        all_cos.append(cos.cpu())
        all_preds.append(pred.cpu())

    all_cos = torch.cat(all_cos)
    all_preds = torch.cat(all_preds)

    cos_mean = all_cos.mean().item()

    # Compute last observation (current date) embedding per sample
    last_obs = torch.zeros_like(dataset.before_embs)
    for i in range(len(dataset)):
        n_valid = int(dataset.masks[i].sum().item())
        if n_valid > 0:
            last_obs[i] = dataset.sequences[i, n_valid - 1]
        else:
            last_obs[i] = dataset.before_embs[i]

    map10, det_f1 = compute_map_at_k(
        all_preds, last_obs, text_embs, text_class_names,
        dataset.class_labels, dataset.is_negative, k=10)

    return cos_mean, map10, det_f1


def train_epoch(model, loader, optimizer, device, mag_weight=0.1, predict_time=False):
    model.train()
    total_loss = 0
    total_cos = 0
    n = 0

    for batch in loader:
        seq = batch["sequence"].to(device)
        mask = batch["mask"].to(device)
        before = batch["before_emb"].to(device)
        offsets = batch["temporal_offsets"].to(device)
        lat = batch["latitude"].to(device)
        lon = batch["longitude"].to(device)
        target = batch["target"].to(device)
        query = batch["query_offset"].to(device)

        if predict_time:
            pred, time_pred = model(seq, mask, before, temporal_offsets=offsets,
                                    query_offset=query, latitude=lat, longitude=lon,
                                    predict_time=True)
        else:
            pred = model(seq, mask, before, temporal_offsets=offsets,
                         query_offset=query, latitude=lat, longitude=lon)

        loss = cosine_loss(pred, target) + mag_weight * magnitude_mse(pred, target)

        if predict_time:
            gt_time = batch["lead_time_days"].to(device)
            pos_mask = gt_time > 0
            if pos_mask.any():
                loss = loss + 0.1 * F.mse_loss(time_pred[pos_mask], gt_time[pos_mask] / 365.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cos += F.cosine_similarity(pred, target, dim=-1).mean().item()
        n += 1

    return total_loss / max(n, 1), total_cos / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS.keys())
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dataset", default=DATASET_FILE)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--predict-time", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--model", default="temporal_ar",
                        choices=["temporal_ar", "lstm", "gru"],
                        help="Backbone: temporal_ar (paper's Arch B) or lstm/gru (RNN baselines).")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="RNN hidden dim (per direction). Only used when --model lstm or gru.")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    gt_mode = exp["gt"]
    use_diffs = exp["diffs"]
    sinusoidal = exp["sinusoidal"]

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    print("=" * 70)
    print(f"V9 TRAINING: {args.experiment}")
    print(f"  GT: {gt_mode}, Diffs: {use_diffs}, Sinusoidal: {sinusoidal}")
    print(f"  predict_time: {args.predict_time}")
    print(f"  Device: {device}, Epochs: {args.epochs}")
    print("=" * 70)

    print("Loading dataset...")
    data = torch.load(args.dataset, weights_only=False, map_location="cpu")
    emb_dim = data["emb_dim"]
    max_seq_len = data["max_seq_len"]

    train_ds = AnticipationDatasetV5(data, "train", gt_mode=gt_mode, augment=True)
    test_ds = AnticipationDatasetV5(data, "test", gt_mode=gt_mode, augment=False)
    oov_ds = AnticipationDatasetV5(data, "oov", gt_mode=gt_mode, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, OOV: {len(oov_ds)}")

    # Text embeddings (match to dataset embedding dim)
    emb_dim = data["emb_dim"] if "emb_dim" in data else data["sequences"].shape[-1]
    text_emb_file = "text_remoteclipft.json" if emb_dim == 768 else "text_rssiglip.json"
    text_embs_path = os.path.join(DATA_DIR, text_emb_file)
    text_embs, text_class_names = None, None
    if os.path.exists(text_embs_path):
        with open(text_embs_path) as f:
            td = json.load(f)
        text_embs = F.normalize(torch.tensor(td["embeddings"]), dim=-1)
        text_class_names = td["classes"]
        print(f"  Text embeddings: {len(text_class_names)} classes")
        if len(text_class_names) != 72:
            print(f"\n  !!!!! WARNING: Expected 72 classes but got {len(text_class_names)}! !!!!!")
            print(f"  !!!!! Check text embeddings file: {text_embs_path} !!!!!\n")

    # Model
    if args.model == "temporal_ar":
        model = TemporalConditionedAR(
            emb_dim=emb_dim, n_heads=args.n_heads, n_layers=args.n_layers,
            ff_dim=args.ff_dim, dropout=args.dropout, max_seq_len=max_seq_len,
            use_diffs=use_diffs, use_learned_input_pos=not sinusoidal,
        ).to(device)
        model_name = "TemporalConditionedAR"
    elif args.model == "lstm":
        model = LSTMPredictorV9(
            emb_dim=emb_dim, n_heads=args.n_heads, hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, ff_dim=args.ff_dim, dropout=args.dropout,
            max_seq_len=max_seq_len,
            use_diffs=use_diffs, use_learned_input_pos=not sinusoidal,
        ).to(device)
        model_name = f"LSTMPredictorV9(hidden={args.hidden_dim})"
    elif args.model == "gru":
        model = GRUPredictorV9(
            emb_dim=emb_dim, n_heads=args.n_heads, hidden_dim=args.hidden_dim,
            n_layers=args.n_layers, ff_dim=args.ff_dim, dropout=args.dropout,
            max_seq_len=max_seq_len,
            use_diffs=use_diffs, use_learned_input_pos=not sinusoidal,
        ).to(device)
        model_name = f"GRUPredictorV9(hidden={args.hidden_dim})"
    else:
        raise ValueError(f"Unknown --model {args.model}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name}, {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = args.start_epoch
    if args.resume:
        print(f"  Resuming from {args.resume}...")
        model.load_state_dict(torch.load(args.resume, weights_only=True, map_location=device),
                              strict=False)
        for _ in range(start_epoch - 1):
            scheduler.step()

    run_dir = os.path.join(args.results_dir, args.experiment)
    os.makedirs(run_dir, exist_ok=True)

    best_seen_map = -1
    best_oov_map = -1
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_cos = train_epoch(model, train_loader, optimizer, device,
                                            predict_time=args.predict_time)
        scheduler.step()

        # Quick cos_sim eval every epoch
        test_cos, _, _ = eval_and_map(model, test_ds, device,
                                       text_embs, text_class_names) if text_embs is not None else (0, 0, 0)

        elapsed = time.time() - t0

        log = {"epoch": epoch, "train_loss": train_loss, "train_cos": train_cos,
               "test_cos": test_cos, "elapsed": elapsed}

        # Full mAP eval every epoch
        map_str = ""
        if text_embs is not None:
            t_eval = time.time()
            test_cos, test_map10, det_f1 = eval_and_map(
                model, test_ds, device, text_embs, text_class_names)

            oov_map10 = 0.0
            if len(oov_ds) > 0:
                _, oov_map10, _ = eval_and_map(
                    model, oov_ds, device, text_embs, text_class_names)

            log["test_mAP10"] = test_map10
            log["oov_mAP10"] = oov_map10
            log["det_f1"] = det_f1

            map_str = f" | seen_mAP10={test_map10:.3f} oov_mAP10={oov_map10:.3f} F1={det_f1:.3f}"
            eval_time = time.time() - t_eval

            # Save best by seen mAP
            if test_map10 > best_seen_map:
                best_seen_map = test_map10
                torch.save(model.state_dict(), os.path.join(run_dir, "best_seen.pt"))

            # Save best by OOV mAP
            if oov_map10 > best_oov_map:
                best_oov_map = oov_map10
                torch.save(model.state_dict(), os.path.join(run_dir, "best_oov.pt"))

            map_str += f" [{eval_time:.0f}s]"

        history.append(log)

        # Print every epoch
        if True:
            print(f"  ep {epoch:3d} | loss={train_loss:.4f} cos={train_cos:.3f} "
                  f"| test={test_cos:.3f}{map_str} | {elapsed:.0f}s", flush=True)

        # Checkpoint every epoch
        if True:
            torch.save(model.state_dict(), os.path.join(run_dir, f"epoch_{epoch}.pt"))

    # Final summary
    print(f"\n  Best Seen mAP@10: {best_seen_map:.4f}")
    print(f"  Best OOV mAP@10:  {best_oov_map:.4f}")

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    config = {**vars(args), **exp, "n_params": n_params,
              "best_seen_map10": best_seen_map, "best_oov_map10": best_oov_map}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to {run_dir}/")


if __name__ == "__main__":
    main()
