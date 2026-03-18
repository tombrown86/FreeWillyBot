"""
Final validation — Orchestrate rolling walk-forward for regression strategy.

Runs 12×1m and 6×2m windows with fixed config.
Supports --kill-switch N to test the kill switch.
Outputs: data/backtests_regression/walk_forward_1m.csv, walk_forward_2m.csv
         (and walk_forward_1m_ks.csv, walk_forward_2m_ks.csv when kill switch is active)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"


def run(kill_switch_n: int = 0, kill_switch_pf: float = 0.9, dd_kill: float = 0.0, pause_bars: int = 0) -> int:
    """Orchestrate predict (if needed) + walk-forward. Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    test_pred_path = PREDICTIONS_DIR / "test_predictions.parquet"
    if not test_pred_path.exists():
        logging.info("Test predictions not found. Running predict_regression_test...")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "predict_regression_test.py")],
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            return 1

    from src.walk_forward_regression import run_walk_forward

    suffix = ""
    if kill_switch_n > 0:
        suffix += f"_ks{kill_switch_n}"
    if dd_kill > 0:
        dd_label = str(dd_kill).replace(".", "")
        suffix += f"_dd{dd_label}"

    logging.info(
        "Running walk-forward (kill_switch_n=%d pf=%.2f dd_kill=%.3f pause_bars=%d)",
        kill_switch_n, kill_switch_pf, dd_kill, pause_bars,
    )

    results_1m, results_2m = run_walk_forward(
        pred_path=test_pred_path,
        kill_switch_n=kill_switch_n,
        kill_switch_pf=kill_switch_pf,
        dd_kill=dd_kill,
        pause_bars=pause_bars,
    )

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_1m).to_csv(BACKTESTS_DIR / f"walk_forward_1m{suffix}.csv", index=False)
    pd.DataFrame(results_2m).to_csv(BACKTESTS_DIR / f"walk_forward_2m{suffix}.csv", index=False)

    n_pos_1m = sum(1 for r in results_1m if r["net_return"] > 0)
    n_pos_2m = sum(1 for r in results_2m if r["net_return"] > 0)
    logging.info(
        "Walk-forward complete: %d 1m windows (%d/%d positive), %d 2m windows (%d/%d positive)",
        len(results_1m), n_pos_1m, len(results_1m),
        len(results_2m), n_pos_2m, len(results_2m),
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-switch", type=int, default=0, metavar="N",
                        help="Pause trading when last N completed trades PF < kill-pf (0 = disabled)")
    parser.add_argument("--kill-pf", type=float, default=0.9, metavar="PF",
                        help="Kill switch PF threshold (default 0.9)")
    parser.add_argument("--dd-kill", type=float, default=0.0, metavar="DD",
                        help="Pause trading when drawdown from peak exceeds this fraction (e.g. 0.02 = 2%%; 0 = disabled)")
    parser.add_argument("--pause-bars", type=int, default=0, metavar="N",
                        help="Resume trading after N bars following a kill trigger (0 = permanent pause within window)")
    args = parser.parse_args()
    sys.exit(run(kill_switch_n=args.kill_switch, kill_switch_pf=args.kill_pf,
                 dd_kill=args.dd_kill, pause_bars=args.pause_bars))
