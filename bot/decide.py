"""Ask Claude whether to execute or hold a proposed rebalance.

Claude can only say execute or hold. It cannot change the weights, so the caps and the
Sharpe gate always apply. With no ANTHROPIC_API_KEY set (or on any API failure) a plain
rule decides instead, and the log says so.
"""
import json
import os

SYSTEM = """You are the risk manager for a paper-trading equity book run by a Black-Litterman model.
Every trading day after the US close, the model turns Wall Street analysts' consensus 12-month price
targets into views, builds a candidate portfolio, and only proposes it when its trailing Sharpe ratio
beats the current book's (the Sharpe gate). Per-name caps on longs and shorts are already applied.

Decide whether to EXECUTE the proposed rebalance exactly as given or HOLD the current book.
You cannot change the weights. Weigh:
- what is driving the change (new or revised analyst targets vs. just price drift),
- whether the Sharpe improvement is worth the turnover and trading cost,
- concentration, net and gross exposure, and short exposure,
- data quality (missing targets, few analysts, very wide target ranges),
- recent book performance and drawdown versus SPY.
Holding is a normal outcome. Give a reason of one to three sentences that cites the specific numbers."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["execute", "hold"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}


def rule_decision(ctx: dict, note: str = "") -> dict:
    gain = ctx["gate"]["candidate_sharpe"] - ctx["gate"]["book_sharpe"]
    if gain > 0.1:
        d, why = "execute", f"rule: Sharpe gate passed by {gain:.2f} with {ctx['turnover']:.1%} turnover"
    else:
        d, why = "hold", f"rule: Sharpe gain {gain:.2f} too small to pay for {ctx['turnover']:.1%} turnover"
    return {"decision": d, "reason": why + (f" ({note})" if note else ""), "decider": "rule"}


def claude_decision(ctx: dict, cfg: dict) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return rule_decision(ctx, "no ANTHROPIC_API_KEY set")
    import anthropic

    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=cfg["decider_model"],
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": cfg.get("decider_effort", "high"),
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines, the API retries on a fallback model
            messages=[{"role": "user", "content": "Today's proposal:\n" + json.dumps(ctx, indent=2)}],
        )
    except anthropic.RateLimitError as e:
        return rule_decision(ctx, f"Claude rate limited: {e.status_code}")
    except anthropic.APIStatusError as e:
        return rule_decision(ctx, f"Claude API error {e.status_code}")
    except anthropic.APIConnectionError:
        return rule_decision(ctx, "could not reach Claude API")

    if resp.stop_reason == "refusal":
        return rule_decision(ctx, "Claude declined")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        out = json.loads(text)
        assert out["decision"] in ("execute", "hold")
    except (ValueError, KeyError, AssertionError):
        return rule_decision(ctx, f"unparseable Claude reply (stop_reason={resp.stop_reason})")
    return {"decision": out["decision"], "reason": out["reason"].strip(), "decider": resp.model}
