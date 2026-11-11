#!/usr/bin/env python3
"""
Generate RS-SigLIP text embeddings for all class descriptions.

Reads the class descriptions from --input, embeds them with an RS-SigLIP text encoder
deployed on a Vertex AI endpoint (1152-dim), and saves them to --output. The released
data/text_rssiglip.json already contains these embeddings; this script regenerates them.

Usage:
  python -m scripts.text_embeddings_rssiglip --project <gcp-project> --location <region> --endpoint <endpoint-id>
"""

import argparse
import json
import time

from google.cloud import aiplatform

INPUT_FILE = "data/text_rssiglip.json"
OUTPUT_FILE = "data/text_rssiglip_regenerated.json"

BATCH_SIZE = 16
MAX_RETRIES = 5


def embed_texts(endpoint, texts, max_retries=MAX_RETRIES):
    """Embed a batch of text strings via RS-SigLIP text endpoint."""
    for attempt in range(max_retries):
        try:
            prediction = endpoint.predict(instances=texts)
            return prediction.predictions
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project hosting the endpoint")
    parser.add_argument("--location", required=True, help="endpoint region, e.g. us-east4")
    parser.add_argument("--endpoint", required=True, help="Vertex AI endpoint ID of the RS-SigLIP text encoder")
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATING RS-SIGLIP TEXT EMBEDDINGS")
    print("=" * 60)

    # Load class descriptions
    with open(args.input) as f:
        siglip_data = json.load(f)

    classes = siglip_data["classes"]
    descriptions = siglip_data["descriptions"]
    print(f"Loaded {len(classes)} classes from {args.input}")

    # Initialize RS-SigLIP text endpoint
    aiplatform.init(project=args.project, location=args.location)
    endpoint = aiplatform.Endpoint(args.endpoint)
    print(f"Endpoint: {args.endpoint} ({args.project}, {args.location})")

    # Embed all descriptions in batches
    all_embeddings = []
    for i in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i//BATCH_SIZE + 1}/{(len(descriptions) + BATCH_SIZE - 1)//BATCH_SIZE} "
              f"({len(batch)} texts)...")
        embs = embed_texts(endpoint, batch)
        all_embeddings.extend(embs)

    print(f"\nEmbedded {len(all_embeddings)} descriptions")
    print(f"Embedding dim: {len(all_embeddings[0])}")

    # Save
    output = {
        "classes": classes,
        "descriptions": descriptions,
        "embeddings": all_embeddings,
    }
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
