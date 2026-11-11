#!/usr/bin/env python3
"""
Generate RemoteCLIP FT text embeddings using the same 72 classes as RS-SigLIP.

Loads finetuned RemoteCLIP (ViT-L-14, 768-dim), encodes the same class
descriptions used for RS-SigLIP, saves to text_remoteclipft.json.

Usage:
  python -m scripts.text_embeddings_remoteclipft
  python -m scripts.text_embeddings_remoteclipft --weights checkpoints/lora_batch64.pt
"""

import argparse
import json

import torch
import torch.nn.functional as F
import open_clip

INPUT_FILE = "data/text_rssiglip.json"
OUTPUT_FILE = "data/text_remoteclipft.json"
WEIGHTS = "checkpoints/lora_batch64.pt"
MODEL_NAME = "ViT-L-14"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    # Load class list from RS-SigLIP (same 72 classes)
    with open(args.input) as f:
        sig = json.load(f)
    classes = sig["classes"]
    descriptions = sig["descriptions"]
    print(f"Loaded {len(classes)} classes from {args.input}")

    # Load finetuned RemoteCLIP
    print(f"Loading RemoteCLIP from {args.weights}...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="")
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    print("  Model loaded")

    # Embed descriptions
    print("Embedding descriptions...")
    with torch.no_grad():
        tokens = tokenizer(descriptions)
        embs = model.encode_text(tokens)
        embs = F.normalize(embs, dim=-1)

    print(f"  {embs.shape[0]} embeddings, dim={embs.shape[1]}")

    output = {
        "classes": classes,
        "descriptions": descriptions,
        "embeddings": embs.tolist(),
    }
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
