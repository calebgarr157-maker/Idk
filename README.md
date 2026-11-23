# Analyst-target Black-Litterman paper bot

A daily paper-trading bot built on the 3rd-place UChicago 2022 Case 3 strategy
([aardisaputra/uchicago_case3](https://github.com/aardisaputra/uchicago_case3)), moved from simulated
analysts to real stocks and real Wall Street price targets from Yahoo Finance.

## What runs each day (after the close)

1. **Prices**: about 15 months of adjusted closes for the 9 stocks in `bot/config.json`, plus SPY.
2. **Views**: each stock's consensus 12-month analyst target becomes a daily view,
   `log(target / price) / 252`. Every target is appended to `logs/targets.csv`, along with whether it changed.
3. **Black-Litterman**: the prior is market-cap weights. View confidence is the notebook's `tau·PΣP'`,
   scaled down for stocks whose analyst range (high minus low) is wide. Raw weights are scaled to 100% gross.
4. **Caps**: each short is capped at 10% and each long at 30%. Anything clipped off stays in cash.
   The uncapped weights are still logged, so you can see when the raw rule asks for something extreme.
5. **Sharpe gate**: the candidate only goes forward if its trailing 63-day Sharpe beats the current book's,
   the same rule the notebook used. Changes under 2% turnover are skipped.
6. **Claude decides**: if the gate passes, Claude (`claude-fable-5-2`) sees the proposal and answers
   `execute` or `hold` with a reason. It can't change the weights. With no API key, or if the call
   fails, a simple rule decides and the log says so. Everything goes in `logs/decisions.csv`.
7. **Paper book**: trades fill at the close, with 5 bps cost and 0.5%/yr short borrow. Positions are in
   `state/book.json`, fills in `logs/trades.csv`, daily NAV next to SPY in `logs/nav.csv`.
   The book starts at equal weight, as the notebook did.
8. **Weekly vs SPY**: every Friday, `reports/weekly/<date>.md` and `logs/weekly.csv`.

## Usage

```bash
pip install -r requirements.txt
python -m bot.run_daily --dry-run   # compute and print, write nothing
python -m bot.run_daily             # the real daily run (skips if today is already logged)
python -m bot.report                # weekly report on demand
python -m pytest -q tests
```

### Scheduling

A scheduled Claude session follows `bot/DAILY_SESSION.md` each weekday after the close and acts as the decider itself (no API key needed).
Or use `.github/workflows/daily.yml`, which runs at 21:30 UTC on weekdays and commits the logs back.
- Add an `ANTHROPIC_API_KEY` repository secret. Without it, the rule decides instead of Claude.
- GitHub only runs scheduled workflows from the **default branch**, so this starts once the code is merged there.
  You can also trigger it by hand from the Actions tab (`workflow_dispatch`).
- Use only one of the two schedules. Each keeps its own book on its own branch.

## The original notebook

`case3/case3_local.ipynb` is the original Colab notebook run locally on the competition data
(`case3/fetch_case3.sh` downloads it). The changes are listed at the top of the notebook.

## Before going live

This is paper only, and there is no live broker code. Things to settle first:
- Run a few months of paper results and compare them in `logs/weekly.csv`.
- **Net short**: the per-name cap still allows up to 90% net short across 9 names. Set `min_net_exposure` in `bot/config.json` (e.g. `-0.2`) to also limit the book's total net short. It's off by default.
- Paper fills are at the same close used for the decision. Live fills would be the next open or later, with real borrow rates and locate limits.
- Yahoo targets are free consensus data with no SLA. They can lag analyst changes or disappear for a ticker.
