"""V5 Dataset: completion-aligned GT, single target per sample."""

import torch
from torch.utils.data import Dataset


class AnticipationDatasetV5(Dataset):
    """V5 dataset for completion-aligned satellite event anticipation.

    Each sample has:
      - Input: weekly embedding sequence with temporal offsets
      - GT: single target (gt_change or gt_completion)
      - lead_time_days: days until completion (or -1 for negatives)

    Supports both Arch A (fixed horizons) and Arch B (random horizons) via bucket_filter:
      - None: all samples (default, backward compat)
      - "fixed": only lead_time_bucket in [0,1,2,3] (1wk/1mo/3mo/6mo) + negatives
      - "random": only lead_time_bucket == 4 (random lead times) + negatives
    """

    def __init__(self, data_dict, split, gt_mode="gt_change", augment=False,
                 bucket_filter=None):
        """
        Args:
            data_dict: loaded from a dataset built by scripts/build_dataset.py
            split: "train", "test", or "oov"
            gt_mode: "gt_change" or "gt_completion"
            augment: apply noise augmentation (training only)
            bucket_filter: None, "fixed", or "random"
        """
        self.gt_mode = gt_mode
        self.augment = augment
        self.emb_dim = data_dict["emb_dim"]

        # Filter by split
        indices = [i for i, s in enumerate(data_dict["split_labels"]) if s == split]

        # Filter by lead-time bucket
        if bucket_filter is not None:
            buckets = data_dict["lead_time_bucket"]
            neg = data_dict["is_negative"]
            if bucket_filter == "fixed":
                indices = [i for i in indices
                           if neg[i] or buckets[i].item() in (0, 1, 2, 3)]
            elif bucket_filter == "random":
                indices = [i for i in indices
                           if neg[i] or buckets[i].item() == 4]

        self.sequences = data_dict["sequences"][indices]
        self.masks = data_dict["masks"][indices]
        self.temporal_offsets = data_dict["temporal_offsets"][indices]
        self.before_embs = data_dict["before_embs"][indices]
        self.latitudes = data_dict["latitudes"][indices]
        self.longitudes = data_dict["longitudes"][indices]
        self.is_negative = data_dict["is_negative"][indices]
        self.lead_time_days = data_dict["lead_time_days"][indices]
        self.lead_time_bucket = data_dict["lead_time_bucket"][indices]

        # Select GT target
        self.targets = data_dict[gt_mode][indices]  # (N, D)

        self.class_labels = [data_dict["class_labels"][i] for i in indices]
        self.location_ids = [data_dict["location_ids"][i] for i in indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()
        mask = self.masks[idx].clone()

        if self.augment:
            seq, mask = self._augment(seq, mask)

        # Negatives: lead_time_days = -1 in dataset
        # Training: sample random query_offset from [7, 180]
        # Eval: use fixed 30 days (1 month)
        query_offset = self.lead_time_days[idx].clone()
        if self.is_negative[idx] and query_offset.item() < 0:
            if self.augment:  # training
                query_offset = torch.randint(7, 181, (1,)).float().squeeze(0)
            else:  # eval
                query_offset = torch.tensor(30.0)

        return {
            "sequence": seq,
            "mask": mask,
            "temporal_offsets": self.temporal_offsets[idx],
            "before_emb": self.before_embs[idx],
            "latitude": self.latitudes[idx],
            "longitude": self.longitudes[idx],
            "target": self.targets[idx],
            "query_offset": query_offset,
            "lead_time_bucket": self.lead_time_bucket[idx],
            "lead_time_days": self.lead_time_days[idx],
            "is_negative": self.is_negative[idx],
        }

    def _augment(self, seq, mask):
        valid = mask.sum().item()
        if valid == 0:
            return seq, mask

        # Embedding dimension dropout
        if torch.rand(1).item() < 0.5:
            drop_mask = torch.rand(self.emb_dim) > 0.2
            seq = seq * drop_mask.unsqueeze(0)

        # Gaussian jitter on valid positions
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(seq) * 0.05
            seq = seq + noise * mask.unsqueeze(-1).float()

        # Random frame drop
        if valid > 2 and torch.rand(1).item() < 0.3:
            n_drop = max(1, int(valid * (0.1 + 0.2 * torch.rand(1).item())))
            valid_indices = torch.where(mask)[0]
            drop_indices = valid_indices[torch.randperm(len(valid_indices))[:n_drop]]
            mask[drop_indices] = False
            seq[drop_indices] = 0

        return seq, mask
