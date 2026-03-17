"""
Phase 17 — Trade count consistency report.

Load monthly_windows.csv and/or walk_forward_windows.csv, compute n_trades stats,
flag unstable trade frequency (max/min > 3 or std/mean > 0.5).
"""

import logging
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


def _analyze_n_trades(
    n_trades: list[float],
    source: str,
    labels: list[str] | None = None,
) -> None:
    """Compute n_trades stats and flag outliers. labels: optional window labels (e.g. month, fold) for min/max."""
    if not n_trades:
        logging.warning("%s: no n_trades data", source)
        return

    import numpy as np

    arr = np.array(n_trades, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    ratio = max_val / min_val if min_val > 0 else float("inf")

    unstable = []
    if ratio > 3:
        unstable.append("max/min > 3")
    if mean > 0 and std / mean > 0.5:
        unstable.append("std/mean > 0.5")

    logging.info("%s n_trades: mean=%.1f std=%.1f min=%d max=%d max/min=%.2f",
                 source, mean, std, int(min_val), int(max_val), ratio)

    if labels and len(labels) == len(n_trades):
        min_idx = int(np.argmin(arr))
        max_idx = int(np.argmax(arr))
        logging.info("  Min: %s (%d trades)  Max: %s (%d trades) — understand why",
                     labels[min_idx], int(min_val), labels[max_idx], int(max_val))

    if unstable:
        logging.warning("Unstable trade frequency (%s): %s. High variance may indicate weak thresholds or regime dependence.",
                        source, ", ".join(unstable))
    else:
        logging.info("%s: trade frequency stable", source)


def run() -> int:
    """Load validation CSVs and run trade consistency analysis."""
    import csv

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    monthly_path = VALIDATION_DIR / "monthly_windows.csv"
    wf_path = VALIDATION_DIR / "walk_forward_windows.csv"

    found = False
    if monthly_path.exists():
        with open(monthly_path) as f:
            rows = list(csv.DictReader(f))
        n_trades = [float(r["n_trades"]) for r in rows if "n_trades" in r and r["n_trades"]]
        labels = [r.get("month", str(i)) for i, r in enumerate(rows) if "n_trades" in r and r["n_trades"]]
        _analyze_n_trades(n_trades, "monthly_windows", labels=labels if len(labels) == len(n_trades) else None)
        found = True

    if wf_path.exists():
        with open(wf_path) as f:
            rows = list(csv.DictReader(f))
        # Analyze per model_type when present
        if "model_type" in rows[0] if rows else False:
            by_model: dict[str, list[tuple[float, str]]] = {}
            for r in rows:
                if "n_trades" not in r or not r["n_trades"]:
                    continue
                mt = r.get("model_type", "unknown")
                if mt not in by_model:
                    by_model[mt] = []
                fold_label = f"fold{r.get('fold','?')}_{r.get('test_start','')}"
                by_model[mt].append((float(r["n_trades"]), fold_label))
            for model_type, data in by_model.items():
                n_trades_wf = [x[0] for x in data]
                labels_wf = [x[1] for x in data]
                _analyze_n_trades(n_trades_wf, f"walk_forward_windows ({model_type})", labels=labels_wf)
        else:
            n_trades = [float(r["n_trades"]) for r in rows if "n_trades" in r and r["n_trades"]]
            labels = [f"fold{r.get('fold','?')}" for r in rows if "n_trades" in r and r["n_trades"]]
            _analyze_n_trades(n_trades, "walk_forward_windows", labels=labels if len(labels) == len(n_trades) else None)
        found = True

    if not found:
        logging.error("No validation data found. Run run_validate --mode monthly and/or train_price_model --baselines-only first.")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    exit_code = run()
    sys.exit(exit_code)
