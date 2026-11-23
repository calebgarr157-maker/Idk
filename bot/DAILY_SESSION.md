# Daily routine for a scheduled Claude session

Run this from the repository root after the US close on trading days.

1. `git fetch origin claude/stock-portfolio-optimization-0u74xe && git checkout claude/stock-portfolio-optimization-0u74xe && git pull --ff-only`
2. `pip install -q -r requirements.txt`
3. `python -m bot.run_daily --dry-run --show-context`
   - If it prints `... already logged`, stop: today is done.
   - If it prints `NO DECISION NEEDED`, run `python -m bot.run_daily` and go to step 5.
4. If it prints `DECISION NEEDED`, you are the risk manager. Read the `SYSTEM` prompt in
   `bot/decide.py` and judge the printed context by it. Choose `execute` or `hold`. You cannot
   change the weights. Then run:
   `python -m bot.run_daily --decision <execute|hold> --reason "<1-3 sentences citing the numbers>" --decider "claude session"`
5. Commit `logs/ state/ reports/` with the message `daily run <date>` and push to
   `claude/stock-portfolio-optimization-0u74xe`. Change nothing else in the repository.
6. Reply with one line: the date, the decision, the reason, NAV, and return since start vs SPY.
   If a weekly report was written, include its headline numbers.

If Yahoo or the push fails, retry once. If it still fails, say what failed and stop. Never invent data.
