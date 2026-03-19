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

## 2. Crontab entries (match Mac launchd)

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

## 3. Install with the script (optional)

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

## 4. “At load” behaviour on Mac

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

## 5. Logs

Same paths as on the Mac:

- `data/logs/livetick_stdout.log`, `livetick_stderr.log`
- `data/logs/data_refresh_stdout.log`, `data_refresh_stderr.log`
- `data/logs/retrain_stdout.log`, `retrain_stderr.log`

Tail livetick: `tail -f /home/tom/dev/FreeWillyBot/data/logs/livetick_stdout.log`
