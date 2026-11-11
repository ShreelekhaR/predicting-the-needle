#!/usr/bin/env python3
"""Check the environment before building data or training.

Checks package versions, that PyTorch can use the GPU, runs a few training steps of the
model on random data, and (if present) loads the released checkpoints and lists missing
data files. Exits non-zero if anything required fails.

Usage:
  python -m scripts.test_env
  python -m scripts.test_env --cpu-ok     # allow running without a GPU
"""

import argparse
import importlib
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor")

OK, WARN, FAIL = "ok  ", "warn", "FAIL"
failures = []


def report(status, msg):
    print(f"  [{status}] {msg}")
    if status == FAIL:
        failures.append(msg)


def check_packages():
    print("Packages")
    report(OK, f"python {sys.version.split()[0]}")
    for name in ["torch", "numpy", "sklearn", "pandas", "matplotlib"]:
        try:
            mod = importlib.import_module(name)
            report(OK, f"{name} {mod.__version__}")
        except ImportError:
            report(FAIL, f"{name} not installed (pip install -r requirements.txt)")


def check_gpu(cpu_ok):
    import torch
    print("GPU")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()
    driver_msg = next((str(w.message) for w in caught if "driver" in str(w.message).lower()), None)

    if available:
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            report(OK, f"cuda:{i} {props.name} ({props.total_memory / 1e9:.0f} GB)")
        report(OK, f"torch built for CUDA {torch.version.cuda}")
        return torch.device("cuda")

    if driver_msg:
        report(FAIL if not cpu_ok else WARN,
               "PyTorch cannot use the GPU: NVIDIA driver too old for this torch build. "
               "For CUDA 12.4 drivers: pip install \"torch==2.1.2\" \"numpy<2\" "
               "--index-url https://download.pytorch.org/whl/cu121")
    else:
        report(FAIL if not cpu_ok else WARN,
               "no GPU visible to PyTorch; training would run on CPU (about 15x slower)")
    return torch.device("cpu")


def check_model(device):
    import torch
    from anticipation.models.autoregressive_v4 import TemporalConditionedAR
    print("Model (random data)")
    torch.manual_seed(0)
    B, T, D = 32, 53, 1152
    model = TemporalConditionedAR(emb_dim=D, max_seq_len=T, use_diffs=True,
                                  use_learned_input_pos=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    seq = torch.randn(B, T, D, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, 20:] = False
    before = torch.randn(B, D, device=device)
    offsets = torch.arange(T, 0, -1, device=device).float().expand(B, T) * 7
    query = torch.full((B,), 90.0, device=device)
    lat, lon = torch.zeros(B, device=device), torch.zeros(B, device=device)
    target = torch.randn(B, D, device=device)

    def step():
        pred, time_pred = model(seq, mask, before, temporal_offsets=offsets, query_offset=query,
                                latitude=lat, longitude=lon, predict_time=True)
        loss = 1 - torch.nn.functional.cosine_similarity(pred, target).mean() + time_pred.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    try:
        step()  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            loss = step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 10 * 1000
        report(OK, f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
                   f"forward+backward {ms:.0f} ms/batch on {device}, loss {loss:.3f}")
    except Exception as e:  # noqa: BLE001
        report(FAIL, f"training step failed: {type(e).__name__}: {e}")


# release name -> (embedding dim, uses velocity)
CHECKPOINTS = {
    "ps_rssiglip_anticipation.pt": (1152, True),
    "ps_rssiglip_wo_vel_anticipation.pt": (1152, False),
    "s2_rssiglip_anticipation.pt": (1152, True),
    "ps_remoteclipft_anticipation.pt": (768, True),
    "s2_remoteclipft_anticipation.pt": (768, True),
}


def check_checkpoints(ckpt_dir):
    import torch
    from anticipation.models.autoregressive_v4 import TemporalConditionedAR
    print(f"Checkpoints ({ckpt_dir}/)")
    for name, (dim, diffs) in CHECKPOINTS.items():
        path = os.path.join(ckpt_dir, name)
        if not os.path.exists(path):
            report(WARN, f"{name} not found (only needed to reproduce the paper tables)")
            continue
        try:
            model = TemporalConditionedAR(emb_dim=dim, max_seq_len=53, use_diffs=diffs,
                                          use_learned_input_pos=True)
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
            report(OK, f"{name} loads ({dim}-d)")
        except Exception as e:  # noqa: BLE001
            report(FAIL, f"{name} does not load: {type(e).__name__}: {str(e)[:120]}")


DATA_FILES = [
    "data/embeddings/ps_rssiglip_positive_weekly.json",
    "data/embeddings/ps_rssiglip_negative_weekly.json",
    "data/before_after_rssiglip.json",
    "data/text_rssiglip.json",
    "data/labels_positive.csv",
    "data/labels_negative.csv",
    "data/labels_verified.csv",
    "data/completion_dates.json",
    "data/test.pt",
]


def check_data():
    print("Data (PS RS-SigLIP)")
    for f in DATA_FILES:
        if os.path.exists(f):
            report(OK, f"{f} ({os.path.getsize(f) / 1e9:.1f} GB)")
        else:
            report(WARN, f"{f} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-ok", action="store_true", help="do not fail when no GPU is available")
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()

    check_packages()
    if failures:
        print(f"\n{len(failures)} problem(s); fix these first.")
        sys.exit(1)
    device = check_gpu(args.cpu_ok)
    check_model(device)
    check_checkpoints(args.checkpoints)
    check_data()

    print()
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("Environment OK.")


if __name__ == "__main__":
    main()
