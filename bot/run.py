"""Entry point for each cron run (v3).

1. Load state.json (or start a fresh $2,000 account)
2. If no strategy yet: preload 15m history, let Claude design one
   (it picks its own timeframe + markets); rebuild history if it
   chose a different timeframe
3. Fetch every newly closed candle at the strategy's timeframe and
   replay through the engine (trades happen here). Markets seen for
   the first time warm up without trading on stale bars.
4. Self-review every 8 closed trades; if the review changes the
   timeframe, candle history is wiped and rebuilt next runs.
5. Save state.json + docs/data.json for the dashboard
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (MK, TF_MIN, DEFAULT_TF, WARMUP_BARS, REVIEW_EVERY,
                    STALE_REVIEW_H,
                    fresh_state, get_closed_candles, log, now_ms,
                    process_bar, stats, wipe_candles)
import brain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(ROOT, "state.json")
DATA_PATH = os.path.join(ROOT, "docs", "data.json")


def load_state():
    try:
        with open(STATE_PATH) as f:
            return {**fresh_state(), **json.load(f)}, False
    except Exception:
        return fresh_state(), True


def fetch_all(st, tf_name):
    """Fetch new closed candles for every market at tf, oldest first.
    Returns (list of (mkt, bar), set of markets that started empty)."""
    tfm = TF_MIN[tf_name]
    fetched, fresh_markets = [], set()
    for m in MK:
        since = st["last_seen"].get(m) or (now_ms() - WARMUP_BARS * tfm * 60_000)
        if not st["candles"][m]:
            fresh_markets.add(m)
        for bar in get_closed_candles(m, since, tfm):
            fetched.append((m, bar))
    fetched.sort(key=lambda x: x[1]["t"])
    return fetched, fresh_markets


def replay(st, fetched, fresh_markets):
    """Process bars in time order. Markets warming up (first time seen)
    only allow entries on their final 2 bars, so we never trade stale
    history."""
    counts = {}
    for m, _ in fetched:
        counts[m] = counts.get(m, 0) + 1
    seen = {}
    for m, bar in fetched:
        if st["candles"][m] and bar["t"] <= st["candles"][m][-1]["t"]:
            continue
        seen[m] = seen.get(m, 0) + 1
        allow = True
        if m in fresh_markets and seen[m] <= counts[m] - 2:
            allow = False
        process_bar(st, m, bar, allow_entry=allow)


def main():
    st, first_run = load_state()
    if first_run:
        log(st, "Fresh $2,000 paper account initialised.")

    # ---- bootstrap: preload default-tf history, let Claude design ----
    if st["strategy"] is None:
        fetched, fresh = fetch_all(st, DEFAULT_TF)
        # preload without any trading (no strategy yet anyway)
        for m, bar in fetched:
            if st["candles"][m] and bar["t"] <= st["candles"][m][-1]["t"]:
                continue
            process_bar(st, m, bar, allow_entry=False)
        brain.bootstrap(st)
        st["tf"] = st["strategy"]["timeframe"]
        if st["tf"] != DEFAULT_TF:
            wipe_candles(st)
            log(st, f"Timeframe {st['tf']} chosen - rebuilding candle history.", "ai")

    # ---- timeframe change from a past review ----
    if st.get("tf") != st["strategy"]["timeframe"]:
        st["tf"] = st["strategy"]["timeframe"]
        wipe_candles(st)
        log(st, f"Timeframe changed to {st['tf']} - rebuilding candle history.", "ai")

    # ---- fetch + trade ----
    fetched, fresh_markets = fetch_all(st, st["tf"])
    replay(st, fetched, fresh_markets)

    # feed health: crypto is 24/7, so a long silence there means feeds broke
    tfm = TF_MIN[st["tf"]]
    crypto_last = max(st["last_seen"].get(m, 0) for m in ("BTC", "ETH", "SOL"))
    feed_ok = bool(fetched) or (now_ms() - crypto_last < 3 * tfm * 60_000)
    if not feed_ok:
        log(st, "No new candles from any feed for a while - feeds may be down.",
            "warn")

    # ---- self-review when due (trade count OR time limit) ----
    stale_h = STALE_REVIEW_H.get(st["tf"], 48)
    hours_since = (now_ms() - st.get("last_review_t", 0)) / 3600_000
    if False:  # FROZEN: reviews disabled, strategy never mutates
        brain.review(st, idle_hours=round(hours_since))
        if st["strategy"]["timeframe"] != st["tf"]:
            st["tf"] = st["strategy"]["timeframe"]
            wipe_candles(st)
            log(st, f"Timeframe changed to {st['tf']} - rebuilding candle history.",
                "ai")

    # ---- persist full state ----
    with open(STATE_PATH, "w") as f:
        json.dump(st, f)

    # ---- dashboard payload ----
    s = stats(st["trades"])
    data = {
        "updated_at": now_ms(),
        "feed_ok": feed_ok,
        "start_balance": 2000,
        "balance": round(st["balance"], 2),
        "equity": round(st["equity"], 2),
        "timeframe": st.get("tf"),
        "day_anchor": st["day_anchor"],
        "halted": st["halted"],
        "floor_hit": st.get("floor_hit", False),
        "price": st["price"],
        "equity_hist": st["equity_hist"][-700:],
        "positions": st["positions"],
        "trades": st["trades"][:120],
        "strategy": st["strategy"],
        "history": st["history"][:8],
        "log": st["log"][:60],
        "stats": {k: round(v, 2) for k, v in s.items()},
        "trades_since_review": st["trades_since_review"],
        "review_every": REVIEW_EVERY,
    }
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f)

    print(f"OK | equity ${st['equity']:.2f} | {s['n']} trades | "
          f"{len(st['positions'])} open | tf {st.get('tf')} | strategy "
          f"v{st['strategy']['version'] if st['strategy'] else '-'}")


if __name__ == "__main__":
    main()
