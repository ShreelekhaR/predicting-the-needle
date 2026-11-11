#!/usr/bin/env python3
"""
completion_detection — Event-type-aware completion date detection.

Different event classes require different detection strategies:

  PERMANENT   — irreversible state changes (construction, deforestation, etc.)
                Strategy: stable plateau after zero-crossing
                GT = first observation of stable after-state

  TRANSIENT   — temporary changes that revert (flooding, harvest, fire, etc.)
                Strategy: peak M2 score (maximum similarity to after-state)
                GT = observation closest to peak event expression

  CYCLICAL    — repeating seasonal patterns (ploughing, rice paddy, etc.)
                Strategy: first cycle peak M2 score
                GT = first clear expression of the event

Usage:
  python -m scripts.completion_detection
  python -m scripts.completion_detection --output data/completion_dates.json
  python -m scripts.completion_detection --plot-dir plots/completion_debug --n-plots 20
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = "data"
CONSOLIDATED_DIR = os.path.join(DATA_DIR, "embeddings")
WEEKLY_FILE = os.path.join(CONSOLIDATED_DIR, "ps_rssiglip_positive_weekly.json")
BEFORE_AFTER_FILE = os.path.join(DATA_DIR, "before_after_rssiglip.json")
CLASSES_FILE = os.path.join("configs", "positive_classes.json")
DEFAULT_OUTPUT = os.path.join(DATA_DIR, "completion_dates.json")

# ── Event type taxonomy ───────────────────────────────────────────────────────
#
# PERMANENT: stable new state persists after the event.
#   Detection: first stable plateau above MARGIN, held for STABILITY_WINDOW obs.
#   GT embedding: median of observations in the plateau (clean, representative).
#
# TRANSIENT: event is temporary; location returns to baseline afterward.
#   Detection: peak M2 score — the observation most like the after-state.
#   GT embedding: the peak observation itself.
#
# CYCLICAL: repeating seasonal pattern; first-cycle peak.
#   Detection: first local maximum of M2 before any sustained reversal.
#   GT embedding: the first-cycle peak observation.
#
# Classes not listed default to PERMANENT.

EVENT_TYPES = {
    # ── SUDDEN — sharp transition in 1-2 obs, use M2 zero-crossing ──────────
    "Shrubland Wildfire":           "SUDDEN",
    "Snowpack Melt in Mountains":   "SUDDEN",
    "Storm Surge Flooding":         "SUDDEN",
    "Riverine Flood":               "SUDDEN",

    # ── TRANSIENT — temporary peak, use peak M2 ─────────────────────────────
    "Contour Ploughing":            "TRANSIENT",
    "Estuary Sedimentation Change": "TRANSIENT",
    "Fallow Field Tilling":         "TRANSIENT",
    "Harvest 2":                    "TRANSIENT",
    "Leaf Senescence":              "TRANSIENT",
    "Peat Harvesting Operation":    "TRANSIENT",
    "Reservoir Drought":            "TRANSIENT",
    "Rice Paddy Flooding":          "TRANSIENT",
    "Shaped Crop Harvest":          "TRANSIENT",
    "Wheat Harvest":                "TRANSIENT",

    # ── CYCLICAL ─────────────────────────────────────────────────────────────
    "Center Pivot Irrigation Installation": "CYCLICAL",
    "Forest to Pasture Conversion":         "CYCLICAL",
    "Palm Oil Plantation Expansion":        "CYCLICAL",

    # ── PERMANENT (explicit, same as default — listed for clarity) ───────────
    "Airport Construction":                  "PERMANENT",
    "Bridge Construction":                   "PERMANENT",
    "Canal Excavation":                      "PERMANENT",
    "Cemetery Construction":                 "PERMANENT",
    "Coastal Industrial Facility Construction": "PERMANENT",
    "Coastal Resort Development":            "PERMANENT",
    "Dam Construction":                      "PERMANENT",
    "Data Center Construction":              "PERMANENT",
    "Geothermal Power Plant Construction":   "PERMANENT",
    "Golf Course Construction":              "PERMANENT",
    "Greenhouse Complex Buildup":            "PERMANENT",
    "High-Voltage Transmission Line":        "PERMANENT",
    "Highway Construction":                  "PERMANENT",
    "Hospital Wing Addition":                "PERMANENT",
    "Housing Development":                   "PERMANENT",
    "Industrial Site Decommissioning":       "PERMANENT",
    "Industrial Warehouse Development":      "PERMANENT",
    "Landfill Expansion":                    "PERMANENT",
    "Logistics Hub Construction":            "PERMANENT",
    "Mangrove Deforestation":                "PERMANENT",
    "Marsh Restoration":                     "PERMANENT",
    "Military Base Expansion":               "PERMANENT",
    "Mining":                                "PERMANENT",
    "New Island":                            "PERMANENT",
    "New Rail Yard":                         "PERMANENT",
    "New School Construction":               "PERMANENT",
    "Oil and Gas Well Pad Construction":     "PERMANENT",
    "Pier Development":                      "PERMANENT",
    "Pipeline Corridor Clearing":            "PERMANENT",
    "Pit Extraction":                        "PERMANENT",
    "Port Expansion":                        "PERMANENT",
    "Power Plant Construction":              "PERMANENT",
    "Power Plant Construction 2":            "PERMANENT",
    "Power Plant Demolition":                "PERMANENT",
    "Prison Facility Expansion":             "PERMANENT",
    "Quarry Expansion":                      "PERMANENT",
    "Railway Construction":                  "PERMANENT",
    "Recreational Park Development":         "PERMANENT",
    "Refugee Camp Establishment":            "PERMANENT",
    "Reservoir Filling":                     "PERMANENT",
    "Residential Complex Construction":      "PERMANENT",
    "River Channelization":                  "PERMANENT",
    "River Delta Deposition":                "PERMANENT",
    "Runway Extension":                      "PERMANENT",
    "Sports Stadium Erection":               "PERMANENT",
    "Suburban Sprawl":                       "PERMANENT",
    "Tidal Energy Turbine Installation":     "PERMANENT",
    "Timber Salvage Logging":                "PERMANENT",
    "Tropical Deforestation":                "PERMANENT",
    "University Campus Growth":              "PERMANENT",
    "Urban Demolition":                      "PERMANENT",
    "Urban Square Renovation":               "PERMANENT",
    "Utility-Scale Solar Farm":              "PERMANENT",
    "Water Treatment Facility Build":        "PERMANENT",
    "Wind Farm Installation":                "PERMANENT",
}

# Detection parameters
PERMANENT_MARGIN = 0.03       # M2 must exceed this to count as "after-state"
PERMANENT_STABILITY = 4       # consecutive obs must stay above margin
PERMANENT_PLATEAU_WINDOW = 8  # obs after first stable crossing used for GT median

TRANSIENT_MIN_M2 = 0.02       # peak must be at least this high to be valid

CYCLICAL_MIN_M2 = 0.02        # first-cycle peak must exceed this
CYCLICAL_REVERSAL_MARGIN = -0.03  # if M2 drops below this after peak, treat as new cycle


# ── Utilities ────────────────────────────────────────────────────────────────

def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def parse_date(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def detect_outliers(entries: list[dict]) -> set[str]:
    """Flag observations with low similarity to both temporal neighbors (clouds/noise)."""
    if len(entries) < 3:
        return set()
    sorted_e = sorted(entries, key=lambda e: e["date"])
    embs = [np.array(e["embedding"], dtype=np.float32) for e in sorted_e]
    dates = [e["date"] for e in sorted_e]

    sims = []
    for i in range(len(embs)):
        sp = cosine_sim(embs[i], embs[i - 1]) if i > 0 else None
        sn = cosine_sim(embs[i], embs[i + 1]) if i < len(embs) - 1 else None
        sims.append((dates[i], sp, sn))

    all_vals = [s for _, sp, sn in sims for s in [sp, sn] if s is not None]
    if not all_vals:
        return set()
    threshold = max(0.75, np.median(all_vals) - 0.1)

    return {
        d for d, sp, sn in sims
        if sp is not None and sn is not None
        and sp < threshold and sn < threshold
    }


def advance_past_clouds(completion_date: str, series: list[tuple],
                        outlier_dates: set[str]) -> str:
    """If the chosen completion date is cloudy, advance to the next clean observation."""
    if completion_date not in outlier_dates:
        return completion_date
    # Find next non-outlier date after completion
    found = False
    for ds, dt, m2, emb in series:
        if not found:
            if ds == completion_date:
                found = True
            continue
        if ds not in outlier_dates:
            return ds
    return completion_date  # no clean frame found, keep original


def smooth_median(values: list[float], window: int = 5) -> list[float]:
    """Median smoothing preserving endpoints."""
    from scipy.ndimage import median_filter
    arr = np.array(values)
    if len(arr) < 4:
        return values
    w = min(window, len(arr) - 2)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return values
    smoothed = median_filter(arr, size=w, mode='nearest')
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed.tolist()


def build_m2_series(entries: list[dict], before_emb, after_emb,
                    outlier_dates: set[str]) -> list[tuple[str, datetime, float, np.ndarray]]:
    """
    Returns list of (date_str, datetime, m2_score, embedding) sorted by date,
    with outliers removed.
    """
    rows = []
    for e in entries:
        ds = e.get("date", "")
        if ds in outlier_dates:
            continue
        dt = parse_date(ds)
        if dt is None:
            continue
        emb = np.array(e["embedding"], dtype=np.float32)
        m2 = cosine_sim(emb, after_emb) - cosine_sim(emb, before_emb)
        rows.append((ds, dt, m2, emb))
    rows.sort(key=lambda x: x[1])
    return rows


# ── Detection strategies ──────────────────────────────────────────────────────

def detect_permanent(series: list[tuple], margin: float = PERMANENT_MARGIN,
                     stability: int = PERMANENT_STABILITY,
                     plateau_window: int = PERMANENT_PLATEAU_WINDOW) -> dict | None:
    """
    Find first stable plateau: M2 > margin for `stability` consecutive obs.
    GT embedding = median of obs in plateau window (not just the crossing point).
    Returns None if no stable plateau found.
    """
    if len(series) < stability:
        return None

    values = smooth_median([r[2] for r in series])

    for i in range(len(values) - stability + 1):
        window = values[i:i + stability]
        if all(v > margin for v in window):
            # Found stable crossing at index i
            completion_date = series[i][0]
            completion_dt = series[i][1]

            # GT = median embedding of plateau window
            plateau = series[i:i + plateau_window]
            plateau_embs = np.stack([r[3] for r in plateau])
            gt_emb = np.median(plateau_embs, axis=0)
            gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)

            # M2 at plateau start
            m2_at_completion = series[i][2]

            return {
                "completion_date": completion_date,
                "completion_dt": completion_dt,
                "gt_embedding": gt_emb,
                "m2_at_completion": float(m2_at_completion),
                "plateau_start_idx": i,
                "method": "stable_plateau",
            }

    # Fallback: highest sustained M2 if no clean plateau
    values_raw = [r[2] for r in series]
    best_idx = int(np.argmax(values_raw))
    if values_raw[best_idx] > 0:
        gt_emb = series[best_idx][3].copy()
        gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
        return {
            "completion_date": series[best_idx][0],
            "completion_dt": series[best_idx][1],
            "gt_embedding": gt_emb,
            "m2_at_completion": float(values_raw[best_idx]),
            "plateau_start_idx": best_idx,
            "method": "peak_fallback",
        }

    return None


def detect_sudden(series: list[tuple]) -> dict | None:
    """
    For sudden events (wildfire, flood, snowmelt): M2 zero-crossing.
    The transition happens in 1-2 observations — find the first sustained crossing.
    """
    if len(series) < 3:
        return None

    values = smooth_median([r[2] for r in series])

    # First point that crosses above 0 with next point also >= 0
    for i in range(1, len(values) - 1):
        if values[i - 1] <= 0 and values[i] > 0 and values[i + 1] >= 0:
            gt_emb = series[i][3].copy()
            gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
            return {
                "completion_date": series[i][0],
                "completion_dt": series[i][1],
                "gt_embedding": gt_emb,
                "m2_at_completion": float(values[i]),
                "crossing_idx": i,
                "method": "sudden_zero_crossing",
            }

    # Fallback: max derivative (biggest jump)
    max_jump, best_idx = 0, 0
    for i in range(1, len(values)):
        jump = values[i] - values[i - 1]
        if jump > max_jump:
            max_jump, best_idx = jump, i
    if max_jump > 0:
        gt_emb = series[best_idx][3].copy()
        gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
        return {
            "completion_date": series[best_idx][0],
            "completion_dt": series[best_idx][1],
            "gt_embedding": gt_emb,
            "m2_at_completion": float(values[best_idx]),
            "crossing_idx": best_idx,
            "method": "sudden_max_derivative",
        }

    return None


def detect_transient(series: list[tuple], min_m2: float = TRANSIENT_MIN_M2) -> dict | None:
    """
    For transient events: GT is the observation with peak M2 (maximum after-state expression).
    Completion date = date of peak.
    """
    if not series:
        return None

    values = [r[2] for r in series]
    smoothed = smooth_median(values)
    peak_idx = int(np.argmax(smoothed))

    if smoothed[peak_idx] < min_m2:
        return None

    gt_emb = series[peak_idx][3].copy()
    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)

    return {
        "completion_date": series[peak_idx][0],
        "completion_dt": series[peak_idx][1],
        "gt_embedding": gt_emb,
        "m2_at_completion": float(smoothed[peak_idx]),
        "peak_idx": peak_idx,
        "method": "transient_peak",
    }


def detect_cyclical(series: list[tuple],
                    min_m2: float = CYCLICAL_MIN_M2,
                    reversal_margin: float = CYCLICAL_REVERSAL_MARGIN) -> dict | None:
    """
    For cyclical events: GT is the first-cycle peak before any sustained reversal.
    Finds the first local maximum above min_m2, confirms it's the first cycle
    by checking that M2 hasn't already reversed deeply before it.
    """
    if not series:
        return None

    values = [r[2] for r in series]
    smoothed = smooth_median(values)

    # Find all local maxima
    local_maxima = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if smoothed[i] > min_m2:
                local_maxima.append(i)

    if not local_maxima:
        # No local maximum — fall back to global peak
        return detect_transient(series, min_m2)

    # Take the first local maximum
    first_peak_idx = local_maxima[0]

    # Confirm: after the first peak, does M2 fall significantly (reversal)?
    # If yes, this is genuinely cyclical; if no, treat as transient
    post_peak = smoothed[first_peak_idx:]
    has_reversal = any(v < reversal_margin for v in post_peak)

    gt_emb = series[first_peak_idx][3].copy()
    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)

    method = "cyclical_first_peak" if has_reversal else "cyclical_single_peak"

    return {
        "completion_date": series[first_peak_idx][0],
        "completion_dt": series[first_peak_idx][1],
        "gt_embedding": gt_emb,
        "m2_at_completion": float(smoothed[first_peak_idx]),
        "peak_idx": first_peak_idx,
        "has_reversal": has_reversal,
        "method": method,
    }


# ── Hybrid detection methods (data-driven from user labels) ──────────────────

def _detect_changepoint(series, outlier_dates):
    """CP: CUSUM changepoint on M2 series, skip outliers."""
    if len(series) < 5:
        return None
    clean = [(ds, dt, m2, emb) for ds, dt, m2, emb in series if ds not in outlier_dates]
    if len(clean) < 5:
        return None
    m2_arr = np.array([m2 for _, _, m2, _ in clean])
    cusum = np.cumsum(m2_arr - m2_arr.mean())
    max_idx = int(np.argmax(cusum))
    min_before = int(np.argmin(cusum[:max(max_idx, 1)]))
    if max_idx > min_before:
        diffs = np.abs(np.diff(cusum[min_before:max_idx + 1]))
        cp_idx = min_before + int(np.argmax(diffs)) + 1
    else:
        cp_idx = max_idx
    cp_idx = max(1, min(cp_idx, len(clean) - 1))
    gt_emb = clean[cp_idx][3].copy()
    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
    return {
        "completion_date": clean[cp_idx][0],
        "completion_dt": clean[cp_idx][1],
        "gt_embedding": gt_emb,
        "m2_at_completion": float(clean[cp_idx][2]),
        "method": "hybrid_changepoint",
    }


def _detect_text_sim(series, entries, event_class, outlier_dates):
    """TX: Peak similarity to the after-text description, smoothed, outlier-rejected."""
    import os as _os, json as _json
    text_file = _os.path.join("data", "text_rssiglip_before_after.json")
    if not _os.path.exists(text_file):
        return None
    with open(text_file) as f:
        td = _json.load(f)
    text_embs = {c: np.array(e, dtype=np.float32) for c, e in zip(td["classes"], td["after_embeddings"])}
    text_emb = text_embs.get(event_class)
    if text_emb is None or not series:
        return None

    clean = [(ds, dt, m2, emb) for ds, dt, m2, emb in series if ds not in outlier_dates]
    if len(clean) < 3:
        return None

    sim_text = [float(cosine_sim(emb, text_emb)) for _, _, _, emb in clean]
    arr = np.array(sim_text)
    w = min(5, len(arr))
    if w % 2 == 0: w -= 1
    if w >= 3:
        from scipy.ndimage import median_filter
        smoothed = median_filter(arr, size=w)
    else:
        smoothed = arr

    best_idx = int(np.argmax(smoothed))
    gt_emb = clean[best_idx][3].copy()
    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
    return {
        "completion_date": clean[best_idx][0],
        "completion_dt": clean[best_idx][1],
        "gt_embedding": gt_emb,
        "m2_at_completion": float(clean[best_idx][2]),
        "method": "hybrid_text_sim",
    }


def _detect_knee(series, outlier_dates):
    """KN: Max deceleration in sim_after curve (where growth rate drops most)."""
    clean = [(ds, dt, m2, emb) for ds, dt, m2, emb in series if ds not in outlier_dates]
    if len(clean) < 6:
        return None

    # Use M2 as proxy for sim_after trend
    vals = np.array([m2 for _, _, m2, _ in clean])
    w = min(5, len(vals))
    if w % 2 == 0: w -= 1
    if w >= 3:
        from scipy.ndimage import median_filter
        smoothed = median_filter(vals, size=w)
    else:
        smoothed = vals

    d1 = np.diff(smoothed)
    d2 = np.diff(d1)
    if len(d2) < 3:
        return None

    start = max(1, len(d2) // 5)
    knee_idx = start + int(np.argmin(d2[start:])) + 1
    knee_idx = min(knee_idx, len(clean) - 1)

    gt_emb = clean[knee_idx][3].copy()
    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
    return {
        "completion_date": clean[knee_idx][0],
        "completion_dt": clean[knee_idx][1],
        "gt_embedding": gt_emb,
        "m2_at_completion": float(clean[knee_idx][2]),
        "method": "hybrid_knee",
    }


def _detect_naive_plateau(series, after_emb, entries, outlier_dates, margin=0.02):
    """NP: First clean date where sim_after >= max - margin."""
    clean = [(ds, dt, m2, emb) for ds, dt, m2, emb in series if ds not in outlier_dates]
    if len(clean) < 2:
        return None

    sim_after = [float(cosine_sim(emb, after_emb)) for _, _, _, emb in clean]
    max_sim = max(sim_after)
    thresh = max_sim - margin

    for i, (ds, dt, m2, emb) in enumerate(clean):
        if sim_after[i] >= thresh:
            gt_emb = emb.copy()
            gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
            return {
                "completion_date": ds,
                "completion_dt": dt,
                "gt_embedding": gt_emb,
                "m2_at_completion": float(m2),
                "method": "hybrid_naive_plateau",
            }
    return None


# ── Main detection dispatch ───────────────────────────────────────────────────

def detect_completion(loc_id: str, entries: list[dict],
                      before_emb, after_emb,
                      event_class: str) -> dict:
    """
    Route to the appropriate detection strategy based on event type.
    Returns a result dict with at minimum:
        completion_date, gt_embedding, m2_at_completion, method, event_type
    """
    event_type = EVENT_TYPES.get(event_class, "PERMANENT")
    outliers = detect_outliers(entries)
    series = build_m2_series(entries, before_emb, after_emb, outliers)

    # Load hybrid method mapping if available
    import os as _os
    _method_file = _os.path.join("configs", "event_completion_methods.json")
    _hybrid_method = None
    if _os.path.exists(_method_file):
        import json as _json
        with open(_method_file) as _f:
            _method_map = _json.load(_f)
        _hybrid_method = _method_map.get(event_class)

    result = None
    if _hybrid_method == "TX":
        result = _detect_text_sim(series, entries, event_class, outliers)
    elif _hybrid_method == "KN":
        result = _detect_knee(series, outliers)
    elif _hybrid_method == "NP":
        result = _detect_naive_plateau(series, after_emb, entries, outliers)
    elif _hybrid_method == "CP":
        result = _detect_changepoint(series, outliers)
    elif event_type == "PERMANENT":
        result = detect_permanent(series)
    elif event_type == "TRANSIENT":
        result = detect_transient(series)
    elif event_type == "SUDDEN":
        result = detect_sudden(series)
    elif event_type == "CYCLICAL":
        result = detect_cyclical(series)

    # Advance past cloudy frames if the chosen date is an outlier
    if result and result.get("completion_date"):
        # Check against ALL entries (including outliers) for cloud detection
        all_dates_set = detect_outliers(entries)  # outliers from full set
        orig_date = result["completion_date"]
        # Build full series (with outliers included for navigation)
        full_series = build_m2_series(entries, before_emb, after_emb, set())
        clean_date = advance_past_clouds(orig_date, full_series, all_dates_set)
        if clean_date != orig_date:
            result["completion_date"] = clean_date
            result["original_date_cloudy"] = orig_date
            # Update GT embedding to the clean frame
            for ds, dt, m2, emb in full_series:
                if ds == clean_date:
                    gt_emb = emb.copy()
                    gt_emb = gt_emb / (np.linalg.norm(gt_emb) + 1e-8)
                    result["gt_embedding"] = gt_emb
                    break

    if result is None:
        return {
            "completion_date": None,
            "gt_embedding": None,
            "m2_at_completion": None,
            "method": "failed",
            "event_type": event_type,
            "n_obs": len(series),
            "n_outliers": len(outliers),
        }

    return {
        **result,
        "event_type": event_type,
        "n_obs": len(series),
        "n_outliers": len(outliers),
        # strip datetime object — not JSON serializable
        "completion_dt": None,
        # gt_embedding as list for JSON
        "gt_embedding": result["gt_embedding"].tolist() if result.get("gt_embedding") is not None else None,
    }


# ── Plotting (optional) ───────────────────────────────────────────────────────

def plot_location(loc_id: str, entries: list[dict], before_emb, after_emb,
                  result: dict, out_path: str):
    """Save a diagnostic plot for one location."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    outliers = detect_outliers(entries)
    series = build_m2_series(entries, before_emb, after_emb, outliers)
    if not series:
        return

    dates = [r[1] for r in series]
    m2_raw = [r[2] for r in series]
    m2_smooth = smooth_median(m2_raw)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, m2_raw, color="steelblue", alpha=0.4, linewidth=1, label="M2 raw")
    ax.plot(dates, m2_smooth, color="navy", linewidth=2, label="M2 smoothed")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    event_type = result.get("event_type", "?")
    method = result.get("method", "?")

    if result.get("completion_date"):
        cd = parse_date(result["completion_date"])
        if cd:
            ax.axvline(cd, color="red", linewidth=1.5, linestyle="--",
                       label=f"Completion: {result['completion_date']}")

    # Shade before/after
    if result.get("completion_date"):
        cd = parse_date(result["completion_date"])
        if cd and dates:
            ax.axvspan(dates[0], cd, alpha=0.07, color="red")
            ax.axvspan(cd, dates[-1], alpha=0.07, color="green")

    class_name = loc_id.rsplit("_", 1)[0].replace("_", " ")
    ax.set_title(f"{class_name}\n{loc_id[:60]} | type={event_type} | method={method}",
                 fontsize=9)
    ax.set_ylabel("M2 (sim_after − sim_before)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-dir", default=None,
                        help="If set, save diagnostic plots here")
    parser.add_argument("--n-plots", type=int, default=10,
                        help="Max plots per event type (default 10)")
    args = parser.parse_args()

    print("=" * 70)
    print("COMPLETION DATE DETECTION (event-type-aware)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open(WEEKLY_FILE) as f:
        weekly_by_loc = json.load(f)
    print(f"  Weekly locations: {len(weekly_by_loc)}")

    with open(BEFORE_AFTER_FILE) as f:
        ba_raw = json.load(f).get("embeddings", {})
    before_after = {
        loc: {
            "before": np.array(v["before"], dtype=np.float32),
            "after":  np.array(v["after"],  dtype=np.float32),
        }
        for loc, v in ba_raw.items()
        if v.get("before") and v.get("after")
    }
    print(f"  Before/after embeddings: {len(before_after)}")

    with open(CLASSES_FILE) as f:
        positive_classes = set(json.load(f))
    print(f"  Positive classes: {len(positive_classes)}")

    # Build class lookup from loc_id prefix
    # loc_id format: ClassName_s2id  (spaces replaced with underscores)
    def loc_to_class(loc_id: str) -> str | None:
        # Try longest-match against known classes
        for cls in sorted(positive_classes, key=len, reverse=True):
            prefix = cls.replace(" ", "_")
            if loc_id.startswith(prefix + "_"):
                return cls
        return None

    # Run detection
    print("\nDetecting completions...")
    results = {}
    stats = defaultdict(lambda: defaultdict(int))
    plot_counts = defaultdict(int)

    for loc_id, entries in weekly_by_loc.items():
        cls = loc_to_class(loc_id)
        if cls is None:
            continue  # not a positive class

        ba = before_after.get(loc_id)
        if ba is None:
            stats[cls]["no_before_after"] += 1
            continue

        result = detect_completion(
            loc_id=loc_id,
            entries=entries,
            before_emb=ba["before"],
            after_emb=ba["after"],
            event_class=cls,
        )
        result["event_class"] = cls
        result["loc_id"] = loc_id
        results[loc_id] = result

        method = result["method"]
        stats[cls][method] += 1

        # Optional plot
        if args.plot_dir and plot_counts[result["event_type"]] < args.n_plots:
            out_path = os.path.join(
                args.plot_dir, result["event_type"],
                f"{loc_id[:60]}.png"
            )
            plot_location(loc_id, entries, ba["before"], ba["after"], result, out_path)
            plot_counts[result["event_type"]] += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")

    total = len(results)
    has_date = sum(1 for r in results.values() if r["completion_date"])
    failed = total - has_date

    print(f"\nTotal locations processed: {total}")
    print(f"  With completion date: {has_date} ({100*has_date/max(total,1):.1f}%)")
    print(f"  Failed: {failed}")

    by_type = defaultdict(list)
    for r in results.values():
        by_type[r["event_type"]].append(r)

    for etype in ["PERMANENT", "TRANSIENT", "CYCLICAL"]:
        group = by_type[etype]
        if not group:
            continue
        success = sum(1 for r in group if r["completion_date"])
        m2s = [r["m2_at_completion"] for r in group if r.get("m2_at_completion") is not None]
        print(f"\n  {etype}: {len(group)} locations, {success} detected")
        if m2s:
            print(f"    M2 at completion: mean={np.mean(m2s):.3f} "
                  f"median={np.median(m2s):.3f} "
                  f"p10={np.percentile(m2s,10):.3f} "
                  f"p90={np.percentile(m2s,90):.3f}")

    print(f"\nMethod breakdown:")
    method_counts = defaultdict(int)
    for r in results.values():
        method_counts[r["method"]] += 1
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method:30s}: {count}")

    # Save — strip gt_embedding for space, save separately
    print(f"\nSaving to {args.output}...")

    # Separate gt_embeddings from metadata
    gt_embeddings = {}
    output_meta = {}
    for loc_id, r in results.items():
        gt_emb = r.pop("gt_embedding", None)
        if gt_emb is not None:
            gt_embeddings[loc_id] = gt_emb
        output_meta[loc_id] = {k: v for k, v in r.items() if k != "completion_dt"}

    with open(args.output, "w") as f:
        json.dump(output_meta, f, indent=2)

    # Save GT embeddings as tensor
    gt_emb_path = args.output.replace(".json", "_gt_embeddings.pt")
    import torch
    if gt_embeddings:
        loc_ids_list = list(gt_embeddings.keys())
        emb_array = np.stack([gt_embeddings[lid] for lid in loc_ids_list])
        torch.save({
            "loc_ids": loc_ids_list,
            "embeddings": torch.from_numpy(emb_array),
        }, gt_emb_path)
        print(f"GT embeddings saved to {gt_emb_path} ({len(loc_ids_list)} locations)")

    print(f"Metadata saved to {args.output}")
    print("\nDone!")


if __name__ == "__main__":
    main()
    