"""Core paper-trading engine v4 - SCALPER (BTC/ETH/XAU, 1m/5m).

- 10 markets: crypto (Kraken feed), FX / gold / index (Yahoo feed)
- Bot-chosen timeframe: 5m / 15m / 1h (native candles from each feed)
- Realistic per-market spread costs (IC Markets quotes, commission zero)
- FX / metals / index markets close on weekends; engine simply receives
  no new candles then. Crypto runs 24/7.

All signals fire on CLOSED candles. Intrabar exits use the bar's
high/low with stop-before-target (conservative) fill logic.
"""
import time
import requests

# spread = full bid-ask spread as % of price (round trip cost)
MARKETS = {
    "BTC": {"feed": "kraken", "sym": "XBTUSD", "spread": 0.02},
    "ETH": {"feed": "kraken", "sym": "ETHUSD", "spread": 0.16},
    "XAU": {"feed": "yahoo",  "sym": "GC=F",   "spread": 0.011},
}
MK = list(MARKETS.keys())
CRYPTO = ["BTC", "ETH"]

TF_MIN = {"1m": 1, "5m": 5}
DEFAULT_TF = "5m"

START_BAL = 2000.0
REVIEW_EVERY = 8
STALE_REVIEW_H = {"1m": 18, "5m": 24, "15m": 48, "1h": 72}  # force review after this many hours without one
MAX_CANDLES_KEPT = 300
WARMUP_BARS = 70

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) autobot-paper/3.0"}


def now_ms() -> int:
    return int(time.time() * 1000)


def utc_day(t_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t_ms / 1000))


# ---------------- data feeds ----------------
def _kraken(sym: str, tf_min: int, since_ms: int):
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": sym, "interval": tf_min,
                "since": max(0, since_ms // 1000)},
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    res = r.json()["result"]
    rows = next(v for k, v in res.items() if isinstance(v, list))
    return [
        {"t": int(k[0]) * 1000, "o": float(k[1]), "h": float(k[2]),
         "l": float(k[3]), "c": float(k[4])}
        for k in rows
    ]


def _yahoo(sym: str, tf_min: int, since_ms: int):
    interval = {1: "1m", 5: "5m", 15: "15m", 60: "60m"}[tf_min]
    rng = {1: "5d", 5: "5d", 15: "5d", 60: "1mo"}[tf_min]
    r = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
        params={"interval": interval, "range": rng},
        headers=UA, timeout=20,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    out = []
    for i, t in enumerate(ts):
        o, h, l, c = (q["open"][i], q["high"][i], q["low"][i], q["close"][i])
        if None in (o, h, l, c):
            continue
        out.append({"t": int(t) * 1000, "o": float(o), "h": float(h),
                    "l": float(l), "c": float(c)})
    return out


def get_closed_candles(mkt: str, since_ms: int, tf_min: int):
    """New closed candles strictly after since_ms, oldest first.
    Empty list is normal for closed markets (FX weekend)."""
    cfg = MARKETS[mkt]
    try:
        rows = _kraken(cfg["sym"], tf_min, since_ms) if cfg["feed"] == "kraken" \
            else _yahoo(cfg["sym"], tf_min, since_ms)
    except Exception:
        return []
    cutoff = now_ms()
    span = tf_min * 60_000
    rows = [b for b in rows if b["t"] > since_ms and b["t"] + span <= cutoff]
    rows.sort(key=lambda b: b["t"])
    return rows


# ---------------- indicators ----------------
def sma(a, p):
    return sum(a[-p:]) / p if len(a) >= p else None


def rsi(closes, p=14):
    if len(closes) < p + 1:
        return None
    g = l = 0.0
    for i in range(len(closes) - p, len(closes)):
        d = closes[i] - closes[i - 1]
        if d > 0:
            g += d
        else:
            l -= d
    if l == 0:
        return 100.0
    return 100 - 100 / (1 + g / l)


def momentum(a, p):
    return (a[-1] / a[-1 - p] - 1) * 100 if len(a) > p else None


def _stddev(a, p):
    if len(a) < p:
        return None
    w = a[-p:]
    m = sum(w) / p
    return (sum((x - m) ** 2 for x in w) / p) ** 0.5


def atr_pct(bars, p=14):
    if len(bars) < p + 1:
        return None
    trs = []
    for i in range(len(bars) - p, len(bars)):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return (sum(trs) / p) / bars[-1]["c"] * 100


def eval_conditions(conds, bars):
    if not isinstance(conds, list) or not conds or not bars:
        return False
    closes = [b["c"] for b in bars]

    def cmp(x, op, v):
        return x < v if op == "<" else x > v

    for c in conds:
        try:
            ind = c.get("indicator")
            if ind == "rsi":
                r = rsi(closes, int(c.get("period", 14)))
                ok = r is not None and cmp(r, c.get("op", "<"), float(c["value"]))
            elif ind == "sma_cross":
                f = sma(closes, int(c.get("fast", 9)))
                sl = sma(closes, int(c.get("slow", 21)))
                ok = f is not None and sl is not None and (
                    f < sl if c.get("state") == "bearish" else f > sl)
            elif ind == "momentum":
                m = momentum(closes, int(c.get("period", 10)))
                ok = m is not None and cmp(m, c.get("op", ">"), float(c["value"]))
            elif ind == "price_vs_sma":
                sl = sma(closes, int(c.get("period", 50)))
                ok = sl is not None and cmp(closes[-1], c.get("op", ">"), sl)
            elif ind == "atr_pct":
                a = atr_pct(bars, int(c.get("period", 14)))
                ok = a is not None and cmp(a, c.get("op", ">"), float(c["value"]))
            elif ind == "bollinger":
                p = int(c.get("period", 20))
                dev = float(c.get("dev", 2))
                mid = sma(closes, p)
                sd = _stddev(closes, p)
                ok = False
                if mid is not None and sd is not None:
                    px = closes[-1]
                    pos = c.get("position", "below_lower")
                    if pos == "below_lower":
                        ok = px < mid - dev * sd
                    elif pos == "above_upper":
                        ok = px > mid + dev * sd
            elif ind == "breakout":
                p = int(c.get("period", 20))
                ok = False
                if len(bars) >= p + 1:
                    prior = bars[-(p + 1):-1]
                    px = closes[-1]
                    if c.get("direction", "high") == "high":
                        ok = px > max(b["h"] for b in prior)
                    else:
                        ok = px < min(b["l"] for b in prior)
            elif ind == "session":
                hrs = c.get("hours") or [0, 24]
                a, bnd = int(hrs[0]) % 24, int(hrs[1]) % 24
                h = time.gmtime(bars[-1]["t"] / 1000).tm_hour
                ok = (a <= h < bnd) if a <= bnd else (h >= a or h < bnd)
            else:
                ok = False
        except Exception:
            ok = False
        if not ok:
            return False
    return True


# ---------------- state ----------------
def fresh_state():
    t = now_ms()
    return {
        "balance": START_BAL,
        "equity": START_BAL,
        "equity_hist": [[t, START_BAL]],
        "tf": None,
        "candles": {m: [] for m in MK},
        "price": {m: None for m in MK},
        "last_seen": {m: 0 for m in MK},
        "positions": [],
        "trades": [],
        "strategy": None,
        "history": [],
        "log": [],
        "trades_since_review": 0,
        "last_review_t": t,
        "day_anchor": {"day": utc_day(t), "eq": START_BAL},
        "halted": False,
        "floor_hit": False,
        "cooldown": {},
        "seq": 1,
    }


def log(st, msg, kind="info", t=None):
    st["log"] = ([{"t": t or now_ms(), "msg": msg, "kind": kind}] + st["log"])[:120]


def wipe_candles(st):
    st["candles"] = {m: [] for m in MK}
    st["last_seen"] = {m: 0 for m in MK}


# ---------------- trading ----------------
def _fill(mkt, side, mid, is_entry):
    """Longs buy the ask and sell the bid; shorts the reverse.
    Half the spread each side."""
    half = MARKETS.get(mkt, {}).get("spread", 0.1) / 100 / 2
    if side == "long":
        return mid * (1 + half) if is_entry else mid * (1 - half)
    return mid * (1 - half) if is_entry else mid * (1 + half)


def _close(st, pos, price, reason, t):
    fill_px = _fill(pos["market"], pos["side"], price, False)
    pnl = (fill_px - pos["entry"] if pos["side"] == "long"
           else pos["entry"] - fill_px) * pos["units"]
    st["balance"] += pnl
    spread_cost = pos["units"] * price * \
        MARKETS.get(pos["market"], {}).get("spread", 0.1) / 100
    st["trades"] = ([{
        "id": pos["id"], "market": pos["market"], "side": pos["side"],
        "entry": pos["entry"], "exit": fill_px, "units": pos["units"],
        "pnl": round(pnl, 2), "cost": round(spread_cost, 2),
        "t_in": pos["t_in"], "t_out": t,
        "held_min": round((t - pos["t_in"]) / 60_000),
        "exit_reason": reason, "strategy_version": pos["strategy_version"],
    }] + st["trades"])[:500]
    st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
    cd = (st["strategy"] or {}).get("risk", {}).get("cooldownMinutes", 15)
    st["cooldown"][pos["market"]] = t + cd * 60_000
    st["trades_since_review"] += 1
    log(st, f"{reason.upper()} {pos['market']} {pos['side']} -> "
            f"{'+' if pnl >= 0 else ''}${pnl:.2f}",
        "win" if pnl >= 0 else "loss", t)


def _open(st, mkt, side, price, t):
    r = st["strategy"]["risk"]
    entry_px = _fill(mkt, side, price, True)  # pay half-spread on entry
    risk_usd = st["equity"] * (r["riskPerTradePct"] / 100)
    units = risk_usd / (entry_px * (r["stopLossPct"] / 100))
    units = min(units, st["equity"] * 5 / entry_px)  # 5x notional cap
    sl = entry_px * (1 - r["stopLossPct"] / 100) if side == "long" \
        else entry_px * (1 + r["stopLossPct"] / 100)
    tp = entry_px * (1 + r["takeProfitPct"] / 100) if side == "long" \
        else entry_px * (1 - r["takeProfitPct"] / 100)
    st["seq"] += 1
    st["positions"].append({
        "id": st["seq"], "market": mkt, "side": side, "entry": entry_px,
        "units": units, "sl": sl, "tp": tp,
        "trail": r.get("trailingStopPct", 0) or 0, "best": entry_px,
        "t_in": t, "strategy_version": st["strategy"]["version"],
        "risk_usd": round(risk_usd, 2),
    })
    log(st, f"ENTER {mkt} {side} @ {entry_px:.5g} (incl. spread) | "
            f"SL {sl:.5g} TP {tp:.5g} | risk ${risk_usd:.2f}", "trade", t)


def _mark_equity(st, t, throttle_ms=10 * 60_000):
    open_pnl = 0.0
    for p in st["positions"]:
        px = st["price"].get(p["market"]) or p["entry"]
        open_pnl += (px - p["entry"] if p["side"] == "long"
                     else p["entry"] - px) * p["units"]
    st["equity"] = st["balance"] + open_pnl
    hist = st["equity_hist"]
    if not hist or t - hist[-1][0] > throttle_ms:
        hist.append([t, round(st["equity"], 2)])
        st["equity_hist"] = hist[-800:]


def _daily_guard(st, t):
    d = utc_day(t)
    if d != st["day_anchor"]["day"]:
        st["day_anchor"] = {"day": d, "eq": st["equity"]}
        if st["halted"] and not st["floor_hit"]:
            st["halted"] = False
            log(st, "New UTC day - daily halt lifted.", "ai", t)
    dl = (st["strategy"] or {}).get("risk", {}).get("maxDailyLossPct", 8)
    if not st["halted"] and st["equity"] <= st["day_anchor"]["eq"] * (1 - dl / 100):
        st["halted"] = True
        for pos in list(st["positions"]):
            _close(st, pos, st["price"].get(pos["market"]) or pos["entry"],
                   "halt", t)
        log(st, f"Daily loss limit {dl}% hit - flat until next UTC day.",
            "loss", t)
    # OVERALL FLOOR: 50% max drawdown from starting balance = hard stop
    floor = START_BAL * 0.50
    if not st["floor_hit"] and st["equity"] <= floor:
        st["floor_hit"] = True
        st["halted"] = True
        for pos in list(st["positions"]):
            _close(st, pos, st["price"].get(pos["market"]) or pos["entry"],
                   "floor", t)
        log(st, f"OVERALL 50% FLOOR hit (equity ${st['equity']:.0f} <= "
                f"${floor:.0f}) - desk stopped permanently until reset.",
            "loss", t)


def process_bar(st, mkt, bar, allow_entry=True):
    """One closed bar: exits, equity, guard, then (optionally) entries."""
    t = bar["t"]
    st["price"][mkt] = bar["c"]
    st["candles"][mkt] = (st["candles"][mkt] + [bar])[-MAX_CANDLES_KEPT:]
    st["last_seen"][mkt] = max(st["last_seen"].get(mkt, 0), t)

    pos = next((p for p in st["positions"] if p["market"] == mkt), None)
    if pos:
        if pos["trail"] > 0:
            if pos["side"] == "long" and bar["h"] > pos["best"]:
                pos["best"] = bar["h"]
                pos["sl"] = max(pos["sl"], bar["h"] * (1 - pos["trail"] / 100))
            if pos["side"] == "short" and bar["l"] < pos["best"]:
                pos["best"] = bar["l"]
                pos["sl"] = min(pos["sl"], bar["l"] * (1 + pos["trail"] / 100))
        hit_sl = bar["l"] <= pos["sl"] if pos["side"] == "long" else bar["h"] >= pos["sl"]
        hit_tp = bar["h"] >= pos["tp"] if pos["side"] == "long" else bar["l"] <= pos["tp"]
        max_hold = (st["strategy"] or {}).get("risk", {}).get("maxHoldMinutes", 240)
        time_up = t - pos["t_in"] > max_hold * 60_000
        trail_won = pos["trail"] > 0 and (
            (pos["side"] == "long" and pos["sl"] > pos["entry"]) or
            (pos["side"] == "short" and pos["sl"] < pos["entry"]))
        if hit_sl:
            _close(st, pos, pos["sl"], "trail" if trail_won else "stop", t)
        elif hit_tp:
            _close(st, pos, pos["tp"], "target", t)
        elif time_up:
            _close(st, pos, bar["c"], "time", t)

    _mark_equity(st, t)
    _daily_guard(st, t)

    strat = st["strategy"]
    if (not allow_entry or not strat or st["halted"] or st.get("floor_hit")
            or mkt not in strat["markets"]):
        return
    if len(st["positions"]) >= strat["risk"]["maxOpenPositions"]:
        return
    if any(p["market"] == mkt for p in st["positions"]):
        return
    if st["cooldown"].get(mkt, 0) > t:
        return
    hist = st["candles"][mkt]
    if len(hist) < 55:
        return
    if eval_conditions(strat["longConditions"], hist):
        _open(st, mkt, "long", bar["c"], t)
    elif eval_conditions(strat["shortConditions"], hist):
        _open(st, mkt, "short", bar["c"], t)


def stats(trades):
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    return {
        "n": len(trades),
        "win_rate": 100 * len(wins) / len(trades) if trades else 0,
        "pf": (gw / gl) if gl > 0 else (99 if gw > 0 else 0),
        "avg_win": gw / len(wins) if wins else 0,
        "avg_loss": gl / len(losses) if losses else 0,
        "net": gw - gl,
    }


# ---------------- tournament backtester ----------------
def backtest(strategy, bars_by_mkt, start_equity=START_BAL):
    """Simulate a candidate strategy over recent history. Same fill,
    spread, sizing, and exit logic as live. Returns summary stats."""
    r = strategy["risk"]
    equity = balance = float(start_equity)
    positions = {}
    cooldown = {}
    hist = {m: [] for m in strategy["markets"]}
    trades = []

    merged = []
    for m in strategy["markets"]:
        for bar in bars_by_mkt.get(m, []):
            merged.append((m, bar))
    merged.sort(key=lambda x: x[1]["t"])

    def fill(mkt, side, mid, is_entry):
        half = MARKETS.get(mkt, {}).get("spread", 0.1) / 100 / 2
        if side == "long":
            return mid * (1 + half) if is_entry else mid * (1 - half)
        return mid * (1 - half) if is_entry else mid * (1 + half)

    def close_pos(m, price, t):
        pos = positions.pop(m)
        fx = fill(m, pos["side"], price, False)
        pnl = (fx - pos["entry"] if pos["side"] == "long"
               else pos["entry"] - fx) * pos["units"]
        nonlocal_balance[0] += pnl
        trades.append(pnl)
        cooldown[m] = t + r["cooldownMinutes"] * 60_000

    nonlocal_balance = [balance]

    for m, bar in merged:
        t = bar["t"]
        hist[m].append(bar)
        if len(hist[m]) > MAX_CANDLES_KEPT:
            hist[m] = hist[m][-MAX_CANDLES_KEPT:]
        pos = positions.get(m)
        if pos:
            if pos["trail"] > 0:
                if pos["side"] == "long" and bar["h"] > pos["best"]:
                    pos["best"] = bar["h"]
                    pos["sl"] = max(pos["sl"], bar["h"] * (1 - pos["trail"] / 100))
                if pos["side"] == "short" and bar["l"] < pos["best"]:
                    pos["best"] = bar["l"]
                    pos["sl"] = min(pos["sl"], bar["l"] * (1 + pos["trail"] / 100))
            hit_sl = bar["l"] <= pos["sl"] if pos["side"] == "long" else bar["h"] >= pos["sl"]
            hit_tp = bar["h"] >= pos["tp"] if pos["side"] == "long" else bar["l"] <= pos["tp"]
            time_up = t - pos["t_in"] > r["maxHoldMinutes"] * 60_000
            if hit_sl:
                close_pos(m, pos["sl"], t)
            elif hit_tp:
                close_pos(m, pos["tp"], t)
            elif time_up:
                close_pos(m, bar["c"], t)
        equity = nonlocal_balance[0] + sum(
            ((hist[pm][-1]["c"] if hist.get(pm) else p["entry"]) - p["entry"]
             if p["side"] == "long" else
             p["entry"] - (hist[pm][-1]["c"] if hist.get(pm) else p["entry"]))
            * p["units"] for pm, p in positions.items())
        if m in positions or len(positions) >= r["maxOpenPositions"]:
            continue
        if cooldown.get(m, 0) > t or len(hist[m]) < 55:
            continue
        side = None
        if eval_conditions(strategy["longConditions"], hist[m]):
            side = "long"
        elif eval_conditions(strategy["shortConditions"], hist[m]):
            side = "short"
        if side:
            entry = fill(m, side, bar["c"], True)
            risk_usd = equity * r["riskPerTradePct"] / 100
            units = min(risk_usd / (entry * r["stopLossPct"] / 100),
                        equity * 5 / entry)
            sl = entry * (1 - r["stopLossPct"] / 100) if side == "long" \
                else entry * (1 + r["stopLossPct"] / 100)
            tp = entry * (1 + r["takeProfitPct"] / 100) if side == "long" \
                else entry * (1 - r["takeProfitPct"] / 100)
            positions[m] = {"side": side, "entry": entry, "units": units,
                            "sl": sl, "tp": tp,
                            "trail": r.get("trailingStopPct", 0) or 0,
                            "best": entry, "t_in": t}
    wins = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    gw, gl = sum(wins), abs(sum(losses))
    return {
        "net": round(sum(trades), 2),
        "trades": len(trades),
        "pf": round((gw / gl) if gl > 0 else (99 if gw > 0 else 0), 2),
    }
