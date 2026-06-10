# Demo persistence phase (Jun 2026+)

## Goal

Prove **signal persistence** on real demo fills — not discover new signals.

Run an uninterrupted **3–6 month** demo with only the two validated regression strategies. Do not evaluate on win rate or equity after a handful of trades.

## Active strategies

| Strategy | Demo account | Notes |
|----------|--------------|-------|
| `regression_v1` | 4247810 | Baseline ML regression |
| `regression_v2_trendfilter` | 4243419 | Same model + 4h trend gate |

**Disabled** (do not re-enable during this phase):

- `regression_v2_trendfilter_portfolio_vol`
- `classifier_v1`
- `mean_reversion_v1`
- `session_breakout_v1`

## Metrics that matter

Run weekly (or after any incident):

```bash
.venv/bin/python scripts/demo_persistence_report.py
.venv/bin/python scripts/demo_persistence_report.py --since 2026-06-10
```

Track per strategy:

1. **Trade count** — sample size; ignore conclusions until N is meaningful
2. **Profit factor** — gross wins / gross losses on completed round-trips
3. **Max drawdown** — from demo equity curve at close events
4. **Monthly returns** — calendar-month compounding of trade returns

## What not to do during this phase

- Add COT, Target2, regional macro, extra LLM forecasters
- Retrain or swap models without ending the clean window
- Re-enable disabled strategies “just to compare”
- Judge readiness from win rate or first 10 trades

## New machine setup

```bash
git clone <repo> && cd FreeWillyBot
cp .env.live .env                    # refresh cTrader token if needed
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# .env flags
# RUN_LIVETICK_DEMO_BROKER=1
# PS_CTRADER_ACCOUNT_ID=4247810

.venv/bin/python -m scripts.run_daily_data_refresh --phase-a-only
.venv/bin/python scripts/verify_demo_accounts.py
.venv/bin/python scripts/reset_paper_demo_state.py --close-all-accounts --also-strategy-state

# Install launchd agents (Mac) — livetick + data refresh only; skip retrain
# See scripts/launchd/*.plist.template

.venv/bin/python scripts/demo_persistence_report.py --since $(date -u +%Y-%m-%d)
```

## Data continuity checklist

Before calling the window “clean”, confirm:

- [ ] `run_daily_data_refresh --phase-a-only` runs daily (launchd/cron)
- [ ] Livetick heartbeat fresh (`data/logs/execution/livetick_heartbeat.json`)
- [ ] Feature tail not stale (no repeated `portfolio_stale_bar` in livetick log)
- [ ] Both demo accounts authenticate (`scripts/verify_demo_accounts.py`)
- [ ] No stacked broker positions (`scripts/ctrader_net_position.py <login>`)

## Exit criteria (rough guide)

After **≥90 days** and **≥30 completed trades per strategy** (order-of-magnitude, not a hard gate):

- PF ≥ 1.0 on demo with costs included
- Max DD within backtest/walk-forward envelope
- Majority of months non-negative for v2; v1 may be noisier but not structurally broken

Failure modes → pause demo, diagnose infrastructure (data staleness, broker reconcile), **not** immediate model surgery.
