#!/usr/bin/env python3
"""Extract test split from a dataset into a small standalone file.

Usage:
  python -m scripts.extract_test_split --dataset data/train.pt --output data/test.pt
  python -m scripts.extract_test_split --dataset data/train.pt --output data/test.pt --split test oov
"""

import argparse
import torch
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", nargs="+", default=["test", "oov"])
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    data = torch.load(args.dataset, weights_only=False, map_location="cpu")

    indices = [i for i, s in enumerate(data["split_labels"]) if s in args.split]
    print(f"  {len(indices)} samples in splits {args.split} (of {len(data['split_labels'])} total)")

    out = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == len(data["split_labels"]):
            out[key] = val[indices]
        elif isinstance(val, list) and len(val) == len(data["split_labels"]):
            out[key] = [val[i] for i in indices]
        else:
            out[key] = val

    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
