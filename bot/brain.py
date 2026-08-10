"""The Claude brain v3: picks its own timeframe and markets from a
10-instrument universe with real spread costs, designs all parameters,
and revises everything in self-reviews. Hard risk caps always apply."""
import json
import os
import requests

from engine import (MARKETS, MK, TF_MIN, DEFAULT_TF, rsi, momentum, stats,
                    log, now_ms, START_BAL, backtest, get_closed_candles)

MODEL = os.environ.get("BOT_MODEL", "claude-sonnet-4-6")

SPREAD_TABLE = ", ".join(
    f"{m} {MARKETS[m]['spread']}%" for m in MK)

MARKET_NOTES = (
    "Round-trip spread cost per market: " + SPREAD_TABLE + ". "
    "BTC/ETH trade 24/7; XAU (gold) is CLOSED on weekends - no candles "
    "then, positions can gap. This is a SCALPING desk on 1m or 5m candles: "
    "spread is the enemy - XAU cheapest, BTC cheap, ETH 8x BTC per round "
    "trip - and the expected win must comfortably exceed the spread. "
    "Account rules: 15% max DAILY loss (halts to next day), and a HARD 50% "
    "overall floor - if equity ever hits $1000 the desk stops permanently. "
    "So protect capital: use an atr_pct volatility filter to AVOID trading "
    "dead chop where scalps just pay spread, and keep per-trade risk modest. "
    "Executed by a cron every ~5-15 min, so entries fill on closed candles "
    "with some lag - prefer setups that stay valid for several minutes."
)

SCHEMA_TEXT = """{
 "name": string,
 "timeframe": "1m" | "5m",
 "markets": subset (max 4) of """ + json.dumps(MK) + """,
 "rationale": string (<=40 words),
 "longConditions": [cond,...],
 "shortConditions": [cond,...],
 "risk": {"riskPerTradePct":number,"stopLossPct":number,"takeProfitPct":number,"trailingStopPct":number,"maxHoldMinutes":number,"maxOpenPositions":number,"maxDailyLossPct":number,"cooldownMinutes":number}
}
cond is ONE of:
 {"indicator":"rsi","period":int,"op":"<"|">","value":number}
 {"indicator":"sma_cross","fast":int,"slow":int,"state":"bullish"|"bearish"}
 {"indicator":"momentum","period":int,"op":"<"|">","value":number}
 {"indicator":"price_vs_sma","period":int,"op":"<"|">"}
 {"indicator":"atr_pct","period":int,"op":"<"|">","value":number}   (ATR as % of price - volatility filter)
 {"indicator":"bollinger","period":int,"dev":number,"position":"below_lower"|"above_upper"}
 {"indicator":"breakout","period":int,"direction":"high"|"low"}     (close breaks N-bar high/low)
 {"indicator":"session","hours":[startUTC,endUTC]}                  (only trade in this UTC window)"""


def _clamp(v, lo, hi, d):
    try:
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return d
        return min(hi, max(lo, v))
    except (TypeError, ValueError):
        return d


def sanitize(s, version):
    if not isinstance(s, dict):
        return None
    r = s.get("risk") or {}
    mkts = [m for m in (s.get("markets") or []) if m in MK][:4] or ["BTC"]
    tf = s.get("timeframe")
    if tf not in TF_MIN:
        tf = DEFAULT_TF
    return {
        "name": str(s.get("name", "Unnamed"))[:60],
        "version": version,
        "timeframe": tf,
        "markets": mkts,
        "rationale": str(s.get("rationale", ""))[:300],
        "longConditions": (s.get("longConditions") or [])[:5],
        "shortConditions": (s.get("shortConditions") or [])[:5],
        "risk": {
            "riskPerTradePct": _clamp(r.get("riskPerTradePct"), 0.1, 2, 0.75),
            "stopLossPct": _clamp(r.get("stopLossPct"), 0.15, 5, 0.8),
            "takeProfitPct": _clamp(r.get("takeProfitPct"), 0.2, 10, 1.6),
            "trailingStopPct": _clamp(r.get("trailingStopPct"), 0, 5, 0),
            "maxHoldMinutes": _clamp(r.get("maxHoldMinutes"), 15, 2880, 360),
            "maxOpenPositions": int(_clamp(r.get("maxOpenPositions"), 1, 3, 2)),
            "maxDailyLossPct": _clamp(r.get("maxDailyLossPct"), 0.5, 15, 8),
            "cooldownMinutes": _clamp(r.get("cooldownMinutes"), 0, 720, 30),
        },
    }


FALLBACK = sanitize({
    "name": "Bootstrap trend-follow (offline fallback)",
    "timeframe": "5m",
    "markets": ["BTC"],
    "rationale": "Used because the strategy API call failed. SMA trend with RSI pullback entries on cheap-spread markets.",
    "longConditions": [
        {"indicator": "sma_cross", "fast": 9, "slow": 21, "state": "bullish"},
        {"indicator": "rsi", "period": 14, "op": "<", "value": 55},
    ],
    "shortConditions": [
        {"indicator": "sma_cross", "fast": 9, "slow": 21, "state": "bearish"},
        {"indicator": "rsi", "period": 14, "op": ">", "value": 45},
    ],
    "risk": {"riskPerTradePct": 0.75, "stopLossPct": 0.8, "takeProfitPct": 1.6,
             "trailingStopPct": 0, "maxHoldMinutes": 720, "maxOpenPositions": 2,
             "maxDailyLossPct": 3, "cooldownMinutes": 60},
}, 1)


def ask_claude(prompt):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        for base in (os.path.dirname(os.path.abspath(__file__)),
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
            p = os.path.join(base, "api_key.txt")
            if os.path.exists(p):
                with open(p) as f:
                    key = f.read().strip()
                break
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 3000,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json()["content"]
                   if b.get("type") == "text")
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _valid(strat):
    return strat and (strat["longConditions"] or strat["shortConditions"])


def _snapshot(st):
    lines = []
    for m in MK:
        c = [b["c"] for b in st["candles"][m]]
        if not c:
            lines.append(f"{m}: closed/no data")
            continue
        line = f"{m}: {c[-1]:.5g}"
        r = rsi(c)
        mo = momentum(c, 20)
        if r is not None:
            line += f", RSI14 {r:.1f}"
        if mo is not None:
            line += f", mom20 {mo:.2f}%"
        lines.append(line)
    return "; ".join(lines)


BT_BARS = 260  # backtest window per candidate, in bars of its timeframe


def _bt_data(strat):
    tfm = TF_MIN[strat["timeframe"]]
    since = now_ms() - BT_BARS * tfm * 60_000
    return {m: get_closed_candles(m, since, tfm) for m in strat["markets"]}


def run_tournament(st, raw_candidates, base_version):
    """Sanitize candidates, backtest each on recent history at its own
    timeframe, deploy the best. Returns (winner, results) or (None, [])."""
    results = []
    best = None
    for raw in (raw_candidates or [])[:3]:
        strat = sanitize(raw, base_version)
        if not _valid(strat):
            continue
        try:
            res = backtest(strat, _bt_data(strat))
        except Exception:
            res = {"net": 0, "trades": 0, "pf": 0}
        entry = {"name": strat["name"], "tf": strat["timeframe"],
                 "net": res["net"], "trades": res["trades"],
                 "pf": res["pf"], "winner": False}
        results.append((strat, entry))
        if best is None or (res["net"], res["pf"]) > (
                results[best][1]["net"], results[best][1]["pf"]):
            best = len(results) - 1
    if best is None:
        return None, []
    results[best][1]["winner"] = True
    return results[best][0], [e for _, e in results]


def bootstrap(st):
    # FROZEN MODE: run a fixed v12-approx strategy, no tournament.
    strat = sanitize({
        "name": "v12_approx_BTC_Breakout_5m (FROZEN)",
        "timeframe": "5m",
        "markets": ["BTC"],
        "rationale": "Reconstruction of v12: 5m BTC breakout, TP0.55/SL0.22, trail0.15. Frozen full-sample test - NOT expected profitable (v12 self-reported PF 0.69).",
        "longConditions": [{"indicator": "breakout", "period": 15, "direction": "high"}],
        "shortConditions": [{"indicator": "breakout", "period": 15, "direction": "low"}],
        "risk": {"riskPerTradePct": 0.3, "stopLossPct": 0.22, "takeProfitPct": 0.55,
                 "trailingStopPct": 0.15, "maxHoldMinutes": 60, "maxOpenPositions": 1,
                 "maxDailyLossPct": 15, "cooldownMinutes": 35},
    }, 1)
    st["strategy"] = strat
    st["tf"] = strat["timeframe"]
    st["history"] = [{"version": 1, "t": now_ms(),
                      "analysis": strat["rationale"],
                      "changes": ["FROZEN v12-approx"],
                      "strategy": strat}]
    st["last_review_t"] = now_ms()
    log(st, f'FROZEN strategy online: "{strat["name"]}"', "ai")
except Exception as e:
        st["strategy"] = FALLBACK
        st["history"] = [{"version": 1, "t": now_ms(),
                          "analysis": FALLBACK["rationale"],
                          "changes": [f"Fallback strategy (API/tournament failed: {type(e).__name__})"],
                          "strategy": FALLBACK}]
        st["last_review_t"] = now_ms()
        log(st, "Strategy API unavailable - running built-in fallback.", "warn")


def review(st, idle_hours=None):
    if not st["strategy"]:
        st["trades_since_review"] = 0
        st["last_review_t"] = now_ms()
        return
    idle = not st["trades"] or (idle_hours and st["trades_since_review"] == 0)
    idle_note = ""
    if idle:
        idle_note = (
            f"IDLE REVIEW: 0 closed trades since the last review "
            f"(~{idle_hours or '?'}h ago). Either your entry rules are too "
            f"strict to ever trigger, or your markets were closed. Current "
            f"snapshot: {_snapshot(st)}. Propose candidates whose conditions "
            f"will realistically fire on this data - or justify staying "
            f"selective with looser but still disciplined rules. ")
    s_ = stats(st["trades"])
    recent = "(no trades)" if not st["trades"] else "\n".join(
        f'{t["market"]} {t["side"]} {t["entry"]:.5g}->{t["exit"]:.5g} '
        f'pnl {t["pnl"]:.2f} cost {t.get("cost", 0):.2f} '
        f'({t["exit_reason"]}, {t["held_min"]}m, v{t["strategy_version"]})'
        for t in st["trades"][:20])
    by_reason = {}
    total_cost = 0.0
    for t in st["trades"]:
        by_reason[t["exit_reason"]] = round(
            by_reason.get(t["exit_reason"], 0) + t["pnl"], 2)
        total_cost += t.get("cost", 0)
    cur = dict(st["strategy"])
    cur.pop("version", None)
    prompt = (
        f"You are the self-learning brain of a ${START_BAL:.0f} paper "
        f"account (cron-executed, closed-candle signals). {MARKET_NOTES} "
        f"{idle_note}Review your own journal and name your mistakes. Then propose 2-3 "
        f"GENUINELY DIFFERENT improved candidate strategies (may change "
        f"timeframe, markets, indicators, R:R, everything) - they will be "
        f"backtested and the best deployed. Never increase risk after "
        f"losses.\n\n"
        f"Stats: {s_['n']} trades, WR {s_['win_rate']:.1f}%, PF {s_['pf']:.2f}, "
        f"avgW ${s_['avg_win']:.2f}, avgL ${s_['avg_loss']:.2f}, "
        f"net ${s_['net']:.2f}, total spread paid ${total_cost:.2f}, "
        f"equity ${st['equity']:.2f}. PnL by exit: {json.dumps(by_reason)}.\n"
        f"Current v{st['strategy']['version']}: {json.dumps(cur)}\n"
        f"Recent trades:\n{recent}\n\n"
        f'Respond ONLY raw JSON: {{"analysis":string(<=50 words),'
        f'"mistakes":[...],"changes":[...],"candidates":[<schema>,...]}}\n'
        f'{SCHEMA_TEXT}'
    )
    try:
        out = ask_claude(prompt)
        winner, results = run_tournament(
            st, out.get("candidates"), st["strategy"]["version"] + 1)
        if winner is None:
            raise ValueError("no valid candidates")
        st["strategy"] = winner
        st["history"] = ([{
            "version": winner["version"], "t": now_ms(),
            "analysis": str(out.get("analysis", ""))[:400],
            "mistakes": [str(x) for x in (out.get("mistakes") or [])][:4],
            "changes": [str(x) for x in (out.get("changes") or [])][:5],
            "tournament": results,
            "strategy": winner,
        }] + st["history"])[:20]
        log(st, f"Tournament review -> v{winner['version']} "
                f'"{winner["name"]}" deployed ({winner["timeframe"]}, '
                f'{", ".join(winner["markets"])})', "ai")
    except Exception:
        log(st, "Self-review call failed - keeping current strategy.", "warn")
    st["trades_since_review"] = 0
    st["last_review_t"] = now_ms()
