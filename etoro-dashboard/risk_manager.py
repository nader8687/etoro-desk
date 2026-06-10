"""
Master / portfolio-level risk manager.

The per-bot path (signal -> execution_quality -> position_sizer) optimises every
trade *in isolation*.  Nothing there looks at the COMBINED book, so 30 bots on
two correlated assets (XRP/BTC) can:
  • stack the same directional bet (8 bots all SHORT BTC at once), or
  • hedge each other (trend bots SHORT while mean-reversion bots BUY the same
    asset) — paying the spread twice to net ~flat.

This module is a single gatekeeper called right before a position opens.  Each
bot still generates its own signal; the manager inspects the aggregate and
returns ALLOW / SHRINK / BLOCK.  It adds **no API calls** — equity and per-trade
risk come from the SizeDecision the sizer already produced, and the open book
comes from trade_manager's in-memory snapshot.

Design principles
-----------------
* Default-ON but conservative: caps are wide enough not to disturb a small book;
  they only bite when the fleet genuinely concentrates risk.
* Never blocks an EXIT — only new entries.
* Fail-open on internal error: a bug here must not freeze trading (it logs and
  allows), because the per-trade guards still apply underneath.
* Fully configurable via a [risk] table in instruments.toml (hot-reloaded on
  mtime change, same mechanism as instrument_config).

Correlated clusters
-------------------
Assets are grouped by `exit_profiles.asset_class` (crypto, commodity, stock…).
XRP and BTC both map to "crypto", so they share one cluster and the cluster
caps prevent the fleet from piling the whole book into correlated crypto risk.
"""
from __future__ import annotations

import logging
import threading
import time
try:
    import tomllib
except ModuleNotFoundError:                  # Python < 3.11
    import tomli as tomllib                   # type: ignore[no-redef]
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "instruments.toml"


# ── Tunable limits (conservative defaults; override in instruments.toml [risk]) ─
@dataclass
class RiskLimits:
    enabled:                    bool  = True
    # Fleet-wide
    max_concurrent_positions:   int   = 12     # of N bots, how many may be live at once
    max_gross_exposure_pct:     float = 60.0   # Σ position $ ≤ this % of equity
    max_portfolio_heat_pct:     float = 6.0    # Σ risk-to-stop $ ≤ this % of equity
    # Per correlated cluster (e.g. all crypto)
    max_cluster_gross_pct:      float = 45.0   # Σ position $ in one cluster ≤ % equity
    max_cluster_net_pct:        float = 25.0   # |long$ − short$| in cluster ≤ % equity
    max_same_dir_per_cluster:   int   = 6      # count of same-direction positions / cluster
    # Per single asset
    max_positions_per_asset:    int   = 4      # how many bots may hold the same asset
    block_internal_hedge:       bool  = False  # don't open opposite a larger same-asset net
    # Drawdown kill-switch (halts NEW entries only)
    daily_drawdown_halt_pct:    float = 5.0    # halt if today's realised P&L ≤ −% of equity
    # Sizing floor (mirror of position_sizer.MIN_TRADE_USD; a shrink below this denies)
    min_trade_usd:              float = 200.0

    notes: list[str] = field(default_factory=list)


_cfg_lock = threading.Lock()
_cached_limits: Optional[RiskLimits] = None
_cached_mtime: float = -1.0


def load_limits() -> RiskLimits:
    """Parse the [risk] table from instruments.toml, cached by file mtime.

    Unknown / missing keys fall back to the conservative dataclass defaults, so
    a config without a [risk] section behaves exactly as the defaults intend.
    """
    global _cached_limits, _cached_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = -1.0
    with _cfg_lock:
        if _cached_limits is not None and mtime == _cached_mtime:
            return _cached_limits
    limits = RiskLimits()
    try:
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        sec = data.get("risk", {}) or {}
        for fld in RiskLimits.__dataclass_fields__:
            if fld == "notes":
                continue
            if fld in sec:
                cur = getattr(limits, fld)
                setattr(limits, fld, type(cur)(sec[fld]))
    except Exception as exc:
        log.warning("risk_manager: could not load [risk] config (%s) — using defaults", exc)
    # Mirror the sizer's MIN_TRADE_USD if available, so the two never disagree.
    try:
        import position_sizer
        limits.min_trade_usd = float(position_sizer.min_trade_usd())
    except Exception:
        pass
    # UI-editable overrides (Settings tab, persisted on data volume).
    try:
        import user_settings
        if user_settings.has_saved_risk():
            us = user_settings.risk_settings()
            for fld in RiskLimits.__dataclass_fields__:
                if fld == "notes":
                    continue
                setattr(limits, fld, getattr(us, fld))
    except Exception as exc:
        log.debug("risk_manager: user_settings overlay skipped: %s", exc)
    with _cfg_lock:
        _cached_limits = limits
        _cached_mtime = mtime
    return limits


def invalidate_limits_cache() -> None:
    global _cached_limits, _cached_mtime
    with _cfg_lock:
        _cached_limits = None
        _cached_mtime = -1.0


# ── Decision object ───────────────────────────────────────────────────────────
@dataclass
class RiskDecision:
    allow:   bool
    amount:  float          # approved $ amount (may be < requested = shrunk; 0 = blocked)
    reason:  str
    capped_by: str = ""     # which limit bound the size ("" = unconstrained)
    shrunk:  bool = False

    @property
    def blocked(self) -> bool:
        return not self.allow or self.amount <= 0


def _cluster_of(instrument_label: str) -> str:
    try:
        import exit_profiles
        return exit_profiles.asset_class(instrument_label)
    except Exception:
        return "unknown"


def _norm_dir(direction: str) -> str:
    d = (direction or "").upper()
    if d in ("BUY", "LONG"):
        return "LONG"
    if d in ("SELL", "SHORT"):
        return "SHORT"
    return d


# ── Portfolio snapshot (read-only; for the gate, UI, and analytics) ───────────
@dataclass
class PortfolioState:
    equity:          float
    n_open:          int
    gross_usd:       float
    heat_usd:        float
    by_asset:        dict           # iid -> {"label","gross","net","count","long","short"}
    by_cluster:      dict           # cluster -> {"gross","net","long","short","count"}


def portfolio_state(equity: float) -> PortfolioState:
    """Aggregate the live open book (no API calls — uses trade_manager snapshot)."""
    import trade_manager
    by_asset: dict = {}
    by_cluster: dict = {}
    gross = 0.0
    heat = 0.0
    open_trades = []
    try:
        # Legacy shadow trades (pre-advisory conversion) hold no real exposure.
        open_trades = [
            t for t in trade_manager.get_all_open()
            if not getattr(t, "shadow", False)
        ]
    except Exception as exc:
        log.warning("risk_manager: get_all_open failed: %s", exc)

    for t in open_trades:
        amt = float(getattr(t, "trade_amount", 0.0) or 0.0)
        if amt <= 0:
            continue
        d = _norm_dir(getattr(t, "direction", ""))
        signed = amt if d == "LONG" else -amt
        gross += amt
        # heat = distance to stop × notional (risk-to-stop in $)
        entry = float(getattr(t, "entry_price", 0.0) or 0.0)
        stop = float(getattr(t, "stop_loss_price", 0.0) or 0.0)
        if entry > 0 and stop > 0:
            heat += abs(entry - stop) / entry * amt

        iid = getattr(t, "instrument_id", 0)
        a = by_asset.setdefault(iid, {
            "label": getattr(t, "instrument_label", ""),
            "gross": 0.0, "net": 0.0, "count": 0, "long": 0, "short": 0,
        })
        a["gross"] += amt
        a["net"] += signed
        a["count"] += 1
        a["long" if d == "LONG" else "short"] += 1

        cl = _cluster_of(getattr(t, "instrument_label", ""))
        c = by_cluster.setdefault(cl, {
            "gross": 0.0, "net": 0.0, "long": 0, "short": 0, "count": 0,
        })
        c["gross"] += amt
        c["net"] += signed
        c["count"] += 1
        c["long" if d == "LONG" else "short"] += 1

    return PortfolioState(
        equity=float(equity or 0.0), n_open=len([t for t in open_trades]),
        gross_usd=gross, heat_usd=heat, by_asset=by_asset, by_cluster=by_cluster,
    )


# ── Drawdown kill-switch ──────────────────────────────────────────────────────
def _today_realised_pnl() -> float:
    """Sum of today's (UTC) realised bot P&L from the journal. 0 on any error."""
    try:
        import trade_journal
        rows = trade_journal.closed_records()
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(
            float(r.get("pnl_dollars") or 0.0)
            for r in rows
            if str(r.get("ts", ""))[:10] == today and (r.get("bot_id") or "").strip()
        )
    except Exception:
        return 0.0


# ── The gate ──────────────────────────────────────────────────────────────────
def check_new_trade(
    *,
    direction: str,
    amount: float,
    risk_dollars: float,
    equity: float,
    instrument_id: int,
    instrument_label: str,
    strategy: str = "",
    bot_id: str = "",
) -> RiskDecision:
    """Approve / shrink / block a NEW position against portfolio-level limits.

    `amount` and `risk_dollars` are the sizer's output for this trade; `equity`
    is the account equity it already fetched.  Returns a RiskDecision whose
    `amount` is what the caller should actually open (0 = skip).
    """
    limits = load_limits()
    if not limits.enabled:
        return RiskDecision(True, amount, "risk manager disabled")
    if amount <= 0:
        return RiskDecision(False, 0.0, "no size")

    # Fail-open: any internal error allows the trade (per-trade guards still apply).
    try:
        if equity <= 0:
            return RiskDecision(True, amount, "equity unknown — risk manager passes")

        d = _norm_dir(direction)
        ps = portfolio_state(equity)
        cl = _cluster_of(instrument_label)
        cluster = ps.by_cluster.get(cl, {"gross": 0.0, "net": 0.0, "long": 0, "short": 0, "count": 0})
        asset = ps.by_asset.get(instrument_id, {"gross": 0.0, "net": 0.0, "count": 0, "long": 0, "short": 0})

        eq = equity
        signed = amount if d == "LONG" else -amount

        # ── Hard BLOCKS (cannot be fixed by shrinking) ────────────────────────
        # 1. Drawdown kill-switch
        dd_halt = -abs(limits.daily_drawdown_halt_pct) / 100.0 * eq
        today_pnl = _today_realised_pnl()
        if today_pnl <= dd_halt:
            return RiskDecision(
                False, 0.0,
                f"drawdown kill-switch: today P&L ${today_pnl:,.0f} ≤ "
                f"−{limits.daily_drawdown_halt_pct:.0f}% equity (${dd_halt:,.0f})",
                capped_by="drawdown_halt",
            )
        # 2. Max concurrent positions
        if ps.n_open >= limits.max_concurrent_positions:
            return RiskDecision(
                False, 0.0,
                f"max concurrent positions reached ({ps.n_open}/{limits.max_concurrent_positions})",
                capped_by="max_concurrent",
            )
        # 3. Max positions per asset
        if asset["count"] >= limits.max_positions_per_asset:
            return RiskDecision(
                False, 0.0,
                f"max positions on {instrument_label} reached "
                f"({asset['count']}/{limits.max_positions_per_asset})",
                capped_by="max_per_asset",
            )
        # 4. Same-direction crowding in the cluster
        same_dir_count = cluster["long"] if d == "LONG" else cluster["short"]
        if same_dir_count >= limits.max_same_dir_per_cluster:
            return RiskDecision(
                False, 0.0,
                f"too many {d} positions in {cl} cluster "
                f"({same_dir_count}/{limits.max_same_dir_per_cluster}) — correlated stacking",
                capped_by="cluster_same_dir",
            )
        # 5. Anti-internal-hedge: don't open opposite a larger same-asset net
        if limits.block_internal_hedge and asset["net"] != 0:
            opposes = (d == "LONG" and asset["net"] < 0) or (d == "SHORT" and asset["net"] > 0)
            if opposes and abs(asset["net"]) >= amount:
                return RiskDecision(
                    False, 0.0,
                    f"internal hedge blocked: {d} {amount:,.0f} opposes existing "
                    f"net ${asset['net']:,.0f} on {instrument_label}",
                    capped_by="internal_hedge",
                )

        # ── SHRINKABLE caps: reduce `amount` to the tightest headroom ─────────
        capped_by = ""
        # a. Gross exposure (fleet)
        gross_cap = limits.max_gross_exposure_pct / 100.0 * eq
        room = gross_cap - ps.gross_usd
        if amount > room:
            amount = room
            capped_by = f"gross exposure {limits.max_gross_exposure_pct:.0f}% equity"
        # b. Cluster gross
        cl_gross_cap = limits.max_cluster_gross_pct / 100.0 * eq
        room = cl_gross_cap - cluster["gross"]
        if amount > room:
            amount = room
            capped_by = f"{cl} cluster gross {limits.max_cluster_gross_pct:.0f}% equity"
        # c. Cluster net directional — only the same-direction side is constrained
        net_cap = limits.max_cluster_net_pct / 100.0 * eq
        projected_net = cluster["net"] + signed
        if abs(projected_net) > net_cap:
            # how much of THIS trade keeps |net| within the cap
            allowed = net_cap - abs(cluster["net"]) if (cluster["net"] * signed) >= 0 else amount
            allowed = max(0.0, allowed)
            if allowed < amount:
                amount = allowed
                capped_by = f"{cl} cluster net {limits.max_cluster_net_pct:.0f}% equity"
        # d. Portfolio heat
        heat_cap = limits.max_portfolio_heat_pct / 100.0 * eq
        # incremental heat for this trade ≈ risk_dollars (already risk-to-stop $)
        room_heat = heat_cap - ps.heat_usd
        if risk_dollars > 0 and room_heat < risk_dollars:
            # scale amount down proportionally to fit remaining heat
            scale = max(0.0, room_heat) / risk_dollars
            new_amt = amount * scale
            if new_amt < amount:
                amount = new_amt
                capped_by = f"portfolio heat {limits.max_portfolio_heat_pct:.0f}% equity"

        amount = round(max(0.0, amount), 2)
        if amount < limits.min_trade_usd:
            return RiskDecision(
                False, 0.0,
                f"shrunk below min trade ${limits.min_trade_usd:,.0f} by {capped_by or 'limits'} "
                f"(gross ${ps.gross_usd:,.0f}, heat ${ps.heat_usd:,.0f}, open {ps.n_open})",
                capped_by=capped_by or "min_trade",
            )

        shrunk = bool(capped_by)
        reason = (
            f"approved ${amount:,.0f}" + (f" (shrunk by {capped_by})" if shrunk else "")
            + f" · book: {ps.n_open} open, gross ${ps.gross_usd:,.0f}, heat ${ps.heat_usd:,.0f}"
        )
        return RiskDecision(True, amount, reason, capped_by=capped_by, shrunk=shrunk)

    except Exception as exc:
        log.warning("risk_manager.check_new_trade failed (%s) — failing open", exc, exc_info=True)
        return RiskDecision(True, amount, f"risk manager error — passed: {exc}")


# End of risk_manager.py

