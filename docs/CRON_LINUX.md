# Cron on Linux (same schedule as Mac launchd)

Use these cron jobs on your Linux machine (e.g. Kubuntu) so FreeWillyBot runs the same schedule as on the MacBook.

**Project path used below:** `/home/tom/dev/FreeWillyBot`. Change it if your path is different.

---

## 1. Prerequisites

- Clone the repo and create a venv:

```bash
cd /home/tom/dev/FreeWillyBot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # edit if needed
```

- Ensure `data/logs` exists so cron can write logs:

```bash
mkdir -p /home/tom/dev/FreeWillyBot/data/logs
```

---

## 2. Source of truth (code changes)

**Do not commit application code only on the Linux box.** Treat this machine as a **deploy target**: develop and commit on your main checkout (e.g. Mac), push to `origin`, then on the server:

```bash
cd /home/tom/dev/FreeWillyBot
git pull
```

If `git pull` complains that local changes would be overwritten, either **stash** (`git stash push -- path/to/file`) or **discard** edits you no longer need (`git restore path/to/file`), then pull again. Prefer fixing the conflict by aligning the server with what is already on `origin`.

Cron and pipelines may **rewrite** tracked files under `data/` (features, models, predictions). That is expected runtime output; it does not replace the rule above for **source** changes (`src/`, `scripts/`, config you intend to version).

---

## 3. Crontab entries (match Mac launchd)

| Mac launchd job        | Schedule              | Cron line |
|------------------------|-----------------------|-----------|
| livetick                | Every 2 min + at load | `*/2 * * * *` |
| data_refresh            | 00:00 daily + at load | `0 0 * * *` |
| retrain                 | 00:30 daily + at load | `30 0 * * *` |

**Full crontab block** — add with `crontab -e`:

```cron
# FreeWillyBot — same as Mac launchd (livetick every 2 min, daily refresh 00:00, daily retrain 00:30)
PROJECT=/home/tom/dev/FreeWillyBot
PY=$PROJECT/.venv/bin/python

# Livetick: every 2 min (signals + paper; auto data refresh when heartbeat/features stale)
*/2 * * * * cd $PROJECT && $PY -m scripts.run_live_tick >> $PROJECT/data/logs/livetick_stdout.log 2>> $PROJECT/data/logs/livetick_stderr.log

# Data refresh: midnight daily (--skip-if-recent 20 = skip if last run < 20h, e.g. after boot)
0 0 * * * cd $PROJECT && $PY -m scripts.run_daily_data_refresh --skip-if-recent 20 >> $PROJECT/data/logs/data_refresh_stdout.log 2>> $PROJECT/data/logs/data_refresh_stderr.log

# Retrain: 00:30 daily, after data refresh (--skip-if-recent 20)
30 0 * * * cd $PROJECT && $PY -m scripts.run_daily_retrain --skip-if-recent 20 >> $PROJECT/data/logs/retrain_stdout.log 2>> $PROJECT/data/logs/retrain_stderr.log
```

The `$PROJECT` and `$PY` variables are set in the same crontab block, so cron expands them when running the jobs.

---

## 4. Install with the script (optional)

From the repo root:

```bash
cd /home/tom/dev/FreeWillyBot
./scripts/install_cron.sh
```

This **appends** the FreeWillyBot cron block to your current crontab. It uses `/home/tom/dev/FreeWillyBot` as the project path by default. To use another path:

```bash
PROJECT_ROOT=/path/to/FreeWillyBot ./scripts/install_cron.sh
```

To only **print** the lines without installing:

```bash
./scripts/install_cron.sh --print
```

---

## 5. “At load” behaviour on Mac

On the Mac, launchd runs each job once at login/boot (RunAtLoad). With cron there is no “at load”; the first run is at the next scheduled time. So:

- **Livetick:** first run at the next even minute (e.g. 10:00, 10:02, …).
- **Data refresh / retrain:** first run at 00:00 and 00:30 that night.

To run once immediately after setting up cron (e.g. data refresh then retrain):

```bash
cd /home/tom/dev/FreeWillyBot
.venv/bin/python -m scripts.run_daily_data_refresh --skip-if-recent 20
.venv/bin/python -m scripts.run_daily_retrain --skip-if-recent 20
.venv/bin/python -m scripts.run_live_tick
```

---

## 6. Logs

Same paths as on the Mac:

- `data/logs/livetick_stdout.log`, `livetick_stderr.log`
- `data/logs/data_refresh_stdout.log`, `data_refresh_stderr.log`
- `data/logs/retrain_stdout.log`, `retrain_stderr.log`

Tail livetick: `tail -f /home/tom/dev/FreeWillyBot/data/logs/livetick_stdout.log`

---

## 7. No signals on the Linux copy

If the Signal log stays empty on the Linux machine, work through these checks.

### 7.1 Confirm cron is running

```bash
crontab -l   # should show the FreeWillyBot livetick line
ls -la /home/tom/dev/FreeWillyBot/data/logs/livetick_*.log   # files should exist and grow every ~2 min
tail -20 /home/tom/dev/FreeWillyBot/data/logs/livetick_stderr.log   # any Python errors?
tail -20 /home/tom/dev/FreeWillyBot/data/logs/livetick_stdout.log   # any "[classifier_v1] signal=..." lines?
```

If the log files are missing or never updated, cron may not be running the job (wrong user, wrong path in crontab, or cron daemon not running).

### 7.2 Run livetick once by hand

This shows errors that cron would hide:

```bash
cd /home/tom/dev/FreeWillyBot
.venv/bin/python -m scripts.run_live_tick
```

Watch for:

- **FileNotFoundError** (e.g. `meta_model.pkl`, `test.csv`, `regression_best.pkl`) → data or models are missing on this machine.
- **"No signal produced"** or **"[regression_v1] No features_regression_core test file"** → data pipeline hasn’t been run here.

### 7.3 Data and models on the Linux box

The Linux copy needs its own data and models; a plain `git pull` does not create `data/processed`, `data/features`, or `data/models`. If you never ran refresh/train on this machine, strategies will fail or return no rows.

**One-time setup on the Linux machine:**

```bash
cd /home/tom/dev/FreeWillyBot
.venv/bin/python -m scripts.run_daily_data_refresh --skip-if-recent 20
.venv/bin/python -m scripts.run_train_regression
.venv/bin/python -m scripts.run_live_tick
```

After that, cron’s livetick job can run every 2 minutes and append to the Signal log. Optionally run retrain so the classifier is up to date:

```bash
.venv/bin/python -m scripts.run_daily_retrain --skip-if-recent 20
```

### 7.4 Copy data from the Mac (alternative)

If you prefer to reuse the Mac’s data instead of re-downloading and retraining on Linux:

1. On the Mac, tar the data (excluding raw if large):  
   `tar -czvf fwb_data.tar.gz -C /path/to/FreeWillyBot data/processed data/features data/features_regression data/features_regression_core data/models data/logs`
2. Copy `fwb_data.tar.gz` to the Linux box.
3. On Linux: `cd /home/tom/dev/FreeWillyBot && tar -xzvf /path/to/fwb_data.tar.gz`

Then run livetick once by hand (6.2) to confirm signals appear.

### 7.5 InconsistentVersionWarning (sklearn pickle)

If you see:

```text
InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.8.0 when using version 1.7.2
```

**Cause:** The model was saved on a machine with Python 3.11+ and scikit-learn 1.8.0, but this Linux box has Python 3.10 (scikit-learn 1.8+ requires Python 3.11+).

**Options:**

**A) Upgrade Python to 3.11+ (recommended):**

```bash
# Remove old venv
rm -rf /home/tom/dev/FreeWillyBot/.venv
# Create new venv with Python 3.11 (install python3.11 if needed: sudo apt install python3.11 python3.11-venv)
python3.11 -m venv /home/tom/dev/FreeWillyBot/.venv
/home/tom/dev/FreeWillyBot/.venv/bin/pip install -r requirements.txt
```

Then retrain so the pickle matches the new sklearn version:
```bash
.venv/bin/python -m scripts.run_daily_retrain --skip-if-recent 0
```

**B) Retrain on Python 3.10 (keeps 1.7.2):**

If you prefer to stay on Python 3.10, retrain the meta model here so the pickle matches your installed scikit-learn 1.7.2:

```bash
.venv/bin/python -m scripts.run_daily_retrain --skip-if-recent 0
```

The warning will stop because the new pickle is from 1.7.2. Note: the model may behave slightly differently due to sklearn version differences, but it should still work.

**C) Ignore the warning:**

The warning is non-fatal; the model may still load and run. Monitor for any actual errors in the logs.

---

## Livetick environment variables

| Variable | Meaning |
|----------|---------|
| `RUN_LIVETICK_DEMO_BROKER=1` | Send orders to the demo broker (same as `--demo-broker`). |
| `RUN_LIVETICK_PARALLEL_PAPER_SIM=1` | With demo broker, also run an **independent dry-run** on each bar and persist books under `{strategy_id}_paper` in `paper_sim_state.json`. Order rows use `strategy_id` like `classifier_v1_paper` and `mode=sim` so you can compare to demo fills (`mode=demo`) over time. |

Example (cron one-liner): demo + parallel paper sim.

```bash
cd $PROJECT && RUN_LIVETICK_DEMO_BROKER=1 RUN_LIVETICK_PARALLEL_PAPER_SIM=1 $PY -m scripts.run_live_tick
```

CLI equivalent: `python -m scripts.run_live_tick --demo-broker --parallel-paper`.

---

## Reset paper state, strategy state, and signal history

From the repo root:

```bash
cd /home/tom/dev/FreeWillyBot
# Paper sim + demo bookkeeping only (equity 1.0, flat, _demo_broker_pos flat)
.venv/bin/python scripts/reset_paper_demo_state.py

# Also reset regression + mean-reversion JSON state files
.venv/bin/python scripts/reset_paper_demo_state.py --also-strategy-state

# Also delete predictions_live.csv and order log CSVs (dashboard signal / order tables start empty)
.venv/bin/python scripts/reset_paper_demo_state.py --signals

# Clear only demo broker order rows from the trade logs (keeps paper/sim rows)
.venv/bin/python scripts/reset_paper_demo_state.py --demo-orders

# Full wipe (common after debugging multi-strategy demo)
.venv/bin/python scripts/reset_paper_demo_state.py --also-strategy-state --signals
```

The next `run_live_tick` run recreates CSV headers when it appends the first row. Using **`--signals`** deletes the whole trade CSVs; **`--demo-orders`** only strips rows with `mode=demo` from `trade_decisions.csv` and `paper_simulation.csv`.
