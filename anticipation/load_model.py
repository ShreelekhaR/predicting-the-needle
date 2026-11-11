"""Load a released checkpoint from this project.

Bundled alongside the NeurIPS 2026 release. Requires the `models/` package
from the project repo on your PYTHONPATH.

    from load_model import load_checkpoint
    model, meta = load_checkpoint("path/to/best.pt")
    print(meta["metrics"])   # → {"max_f1_global": 0.949, ...}
    model.eval()
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch


def _resolve_class(dotted: str):
    module_name, _, class_name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"model_class must be dotted, got {dotted!r}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load a repacked release checkpoint.

    Returns (model, metadata) where metadata is everything except state_dict.
    """
    payload = torch.load(str(path), weights_only=False, map_location=map_location)
    if not isinstance(payload, dict) or "model_class" not in payload:
        raise ValueError(
            f"{path} does not look like a release payload — no 'model_class' key. "
            "Are you loading a raw state_dict? Use torch.load directly for those."
        )

    cls = _resolve_class(payload["model_class"])
    model = cls(**payload["model_kwargs"])
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{' …' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{' …' if len(unexpected) > 5 else ''}")
    model.to(map_location)

    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, meta


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to best.pt")
    args = ap.parse_args()

    model, meta = load_checkpoint(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {meta['model_class']} ({n_params/1e6:.1f}M params)")
    print(json.dumps({k: meta[k] for k in ("metrics", "training", "license", "git_commit")
                      if k in meta}, indent=2))
