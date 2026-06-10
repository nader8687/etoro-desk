"""One-off: reconcile trade_journal.jsonl against eToro's authoritative trade
history.  Removes phantom duplicate close records (same etoro_position_id
closed multiple times by the re-adoption loop) and corrects the surviving
record's exit/P&L to eToro's actual numbers.

Run inside the dashboard container:  python reconcile_journal.py
A timestamped .bak copy of the journal is written next to the original.
"""
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import trade_journal
from etoro_client import EToroClient

JOURNAL = trade_journal.JOURNAL_PATH  # authoritative path used by the app

BOT_REASONS = {"take_profit", "trailing_stop", "stop_loss", "llm"}


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def main() -> None:
    rows = []
    with open(JOURNAL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"journal records: {len(rows)}")

    # Earliest record → history window
    earliest = min((parse_ts(r.get("ts")) for r in rows if parse_ts(r.get("ts"))), default=None)
    min_date = (earliest - timedelta(days=1)) if earliest else (datetime.now(timezone.utc) - timedelta(days=30))

    client = EToroClient(os.environ.get("ETORO_API_KEY", ""), os.environ.get("ETORO_USER_KEY", ""))
    hist = client.get_all_trade_history(min_date.strftime("%Y-%m-%dT%H:%M:%SZ"), demo=True)
    truth = {int(h["positionId"]): h for h in hist if h.get("positionId")}
    print(f"eToro history rows: {len(hist)} (window from {min_date.date()})")

    by_pid = defaultdict(list)
    for i, r in enumerate(rows):
        pid = r.get("etoro_position_id")
        if pid:
            by_pid[int(pid)].append(i)

    drop: set[int] = set()
    corrected = 0
    unmatched_dupes = 0

    for pid, idxs in by_pid.items():
        if len(idxs) < 2:
            continue
        h = truth.get(pid)
        if h is None:
            # Not in the history window — keep only the LAST record (the close
            # after which the position never reappeared); earlier ones were
            # phantom re-adopt closes.
            for i in idxs[:-1] if idxs == sorted(idxs) else sorted(idxs)[:-1]:
                drop.add(i)
            unmatched_dupes += 1
            continue

        close_dt = parse_ts(h.get("closeTimestamp"))

        # Pick the survivor: prefer a bot-reason record within ±120 s of the
        # real close (the close that actually executed), else nearest in time.
        def dist(i):
            t = parse_ts(rows[i].get("exit_time") or rows[i].get("ts"))
            return abs((t - close_dt).total_seconds()) if (t and close_dt) else float("inf")

        near_bot = [i for i in idxs if rows[i].get("reason") in BOT_REASONS and dist(i) <= 120]
        keep = min(near_bot, key=dist) if near_bot else min(idxs, key=dist)

        # Correct the survivor to eToro truth.
        r = rows[keep]
        invest = float(h.get("investment") or r.get("trade_amount") or 0.0)
        net = float(h.get("netProfit") or 0.0)
        open_dt = parse_ts(h.get("openTimestamp"))
        r["exit_price"] = float(h.get("closeRate") or r.get("exit_price") or 0.0)
        r["entry_price"] = float(h.get("openRate") or r.get("entry_price") or 0.0)
        r["pnl_dollars"] = round(net, 4)
        r["pnl_pct"] = round(net / invest * 100.0, 4) if invest else r.get("pnl_pct", 0.0)
        r["win"] = net > 0
        r["trade_amount"] = invest or r.get("trade_amount")
        if close_dt:
            r["exit_time"] = close_dt.isoformat()
            r["ts"] = close_dt.isoformat()
        if open_dt and close_dt:
            r["holding_min"] = round((close_dt - open_dt).total_seconds() / 60.0, 2)
        r["reconciled"] = True
        corrected += 1
        for i in idxs:
            if i != keep:
                drop.add(i)

    kept = [r for i, r in enumerate(rows) if i not in drop]
    print(f"dropped phantom records: {len(drop)} | corrected survivors: {corrected} "
          f"| dupes outside history window: {unmatched_dupes}")
    old_total = sum(float(r.get("pnl_dollars") or 0) for r in rows)
    new_total = sum(float(r.get("pnl_dollars") or 0) for r in kept)
    print(f"realized P&L: {old_total:+.2f} → {new_total:+.2f}  (Δ {new_total - old_total:+.2f})")

    bak = f"{JOURNAL}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(JOURNAL, bak)
    tmp = f"{JOURNAL}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, JOURNAL)
    print(f"journal rewritten ({len(kept)} records) — backup at {bak}")


if __name__ == "__main__":
    main()
