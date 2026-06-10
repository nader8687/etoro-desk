"""Shared HTML tables for Portfolio and History tabs."""
from __future__ import annotations

import html
from collections.abc import Callable

import timez
import ui
from signal_worker import display_asset_name

_label_for_id: Callable[[int | None], str] = lambda iid: str(iid or "—")


def configure_label_resolver(fn: Callable[[int | None], str]) -> None:
    global _label_for_id
    _label_for_id = fn


def fmt_money(val: float | None) -> str:
    return f"${val:+,.2f}" if val is not None else "—"


def fmt_market_price(val: float | None) -> str:
    if val is None:
        return "—"
    av = abs(val)
    if av >= 100:
        return f"{val:,.2f}"
    if av >= 1:
        return f"{val:.2f}"
    return f"{val:.5f}"


def portfolio_units(p: dict) -> float | None:
    units = p.get("units")
    if units:
        return float(units)
    amount = p.get("amount")
    open_rate = p.get("open_rate")
    if amount and open_rate:
        return float(amount) / float(open_rate)
    return None


def parse_api_timestamp(val: str | None) -> str:
    return timez.fmt_iso(val, "%Y-%m-%d %H:%M")


def parse_api_timestamp_short(val: str | None) -> str:
    return timez.fmt_iso(val, "%d %b %H:%M")


def close_method_label(reason: str, strategy: str = "") -> str:
    """Human-readable exit method for the History close-method column."""
    r = (reason or "").strip().lower()
    strat = (strategy or "").strip().lower()
    labels = {
        "stop_loss": "Stop-loss",
        "take_profit": "Take-profit",
        "trailing_stop": "Trailing stop",
        "manual": "Manual",
        "external": "eToro external",
        "etoro": "—",
    }
    if r in labels:
        return labels[r]
    if r == "llm":
        return "LLM exit" if strat == "llm" else "Strategy exit"
    return (reason or "—").replace("_", " ").title() if r else "—"


def close_method_badge_html(reason: str, strategy: str = "") -> str:
    label = close_method_label(reason, strategy)
    if label == "—":
        return '<span style="font-size:0.7rem;color:#888">—</span>'
    colours = {
        "Stop-loss": ("#ff6b6b", "#2a1515"),
        "Take-profit": ("#3dd68c", "#0f2a1c"),
        "Trailing stop": ("#4da6ff", "#0f1a2a"),
        "Strategy exit": ("#c77dff", "#1f152a"),
        "LLM exit": ("#c77dff", "#1f152a"),
        "Manual": ("#aaa", "#2a2e39"),
        "eToro external": ("#ffb347", "#2a2010"),
    }
    fg, bg = colours.get(label, ("#ccc", "#2a2e39"))
    return (
        f'<span style="font-size:0.7rem;color:{fg};background:{bg};'
        f'padding:1px 7px;border-radius:10px;white-space:nowrap">'
        f'{html.escape(label)}</span>'
    )


def history_stats_html(stats: list[tuple[str, str, str | None]]) -> str:
    cells = []
    for label, value, colour in stats:
        col = colour or ui.C_TEXT
        cells.append(
            f'<div class="pf-stat">'
            f'<p class="pf-stat-label">{html.escape(label)}</p>'
            f'<p class="pf-stat-value" style="color:{col}">{html.escape(value)}</p>'
            f"</div>"
        )
    return f'<div class="pf-hist-stats">{"".join(cells)}</div>'


def portfolio_price_change_html(p: dict) -> str:
    if p.get("live_change") is not None or p.get("live_change_pct") is not None:
        ch = p.get("live_change")
        ch_pct = p.get("live_change_pct")
    else:
        ch = p.get("daily_change")
        ch_pct = p.get("daily_change_pct")
    if ch is None and ch_pct is None:
        return ""
    cls = "up" if (ch or 0) >= 0 and (ch_pct or 0) >= 0 else "down"
    if ch is not None and ch < 0:
        cls = "down"
    elif ch_pct is not None and ch_pct < 0:
        cls = "down"
    parts = []
    if ch is not None:
        sign = "+" if ch >= 0 else ""
        parts.append(f"{sign}{ch:.4f}".rstrip("0").rstrip("."))
    if ch_pct is not None:
        sign = "+" if ch_pct >= 0 else ""
        parts.append(f"({sign}{ch_pct:.2f}%)")
    txt = " ".join(parts)
    return f'<span class="pf-chg {cls}">{txt}</span>'


def portfolio_table_html(live_rows: list[dict]) -> str:
    rows_html = []
    for p in live_rows:
        symbol = (p.get("symbol") or "").strip() or str(p.get("instrument_id") or "—")
        name = (p.get("name") or "").strip() or "—"
        direction = (p.get("direction") or "LONG").upper()
        dir_cls = "short" if direction == "SHORT" else "long"
        units = portfolio_units(p)
        units_txt = f"{units:,.4f}".rstrip("0").rstrip(".") if units else "—"
        net_val = p.get("current_value")
        if net_val is None and p.get("amount") is not None:
            net_val = float(p["amount"]) + float(p.get("pnl") or 0)
        pnl = p.get("pnl")
        pct = p.get("pnl_pct")
        pnl_col = ui.pnl_color(pnl)
        pnl_txt = fmt_money(pnl) if pnl is not None else "—"
        pct_txt = f"{pct:+.2f}%" if pct is not None else "—"
        live_tag = (
            f' <span style="color:{ui.C_LIVE};font-size:0.58rem">●</span>'
            if p.get("current_rate") is not None else ""
        )
        chg_html = portfolio_price_change_html(p)
        rows_html.append(
            f"<tr>"
            f'<td class="pf-left"><p class="pf-symbol">{symbol}</p><p class="pf-name">{name}</p></td>'
            f'<td class="pf-right pf-col-price-gap">'
            f'<p class="pf-price">{fmt_market_price(p.get("current_rate"))}{live_tag}</p>'
            f"{chg_html}</td>"
            f'<td class="pf-left pf-col-units-pos"><p class="pf-units pf-units-line">'
            f"{units_txt}"
            f'<span class="pf-dir {dir_cls}">{direction.title()}</span></p></td>'
            f'<td class="pf-right"><p class="pf-val">{fmt_market_price(p.get("open_rate"))}</p></td>'
            f'<td class="pf-right"><p class="pf-pnl" style="color:{pnl_col}">{pnl_txt}</p></td>'
            f'<td class="pf-right"><p class="pf-pnl" style="color:{pnl_col}">{pct_txt}</p></td>'
            f'<td class="pf-right"><p class="pf-val">{"${:,.2f}".format(net_val) if net_val is not None else "—"}</p></td>'
            f"</tr>"
        )
    n = len(live_rows)
    return (
        '<table class="pf-table pf-table-pos">'
        "<colgroup>"
        '<col style="width:18%"><col style="width:13%"><col style="width:13%">'
        '<col style="width:11%"><col style="width:10%"><col style="width:10%">'
        '<col style="width:25%">'
        "</colgroup>"
        "<thead><tr>"
        f'<th class="pf-th pf-th-left">Asset ({n})</th>'
        '<th class="pf-th pf-th-right pf-th-price-gap">Price</th>'
        '<th class="pf-th pf-th-left pf-th-units-pos">Units</th>'
        '<th class="pf-th pf-th-right">Avg. Open</th>'
        '<th class="pf-th pf-th-right">P/L</th>'
        '<th class="pf-th pf-th-right">P/L (%)</th>'
        '<th class="pf-th pf-th-right pf-sorted">Net Value ▾</th>'
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def _owner_chip_html(owner: str) -> str:
    chip_style = (
        "font-size:0.72rem;padding:2px 7px;border-radius:10px;"
        "white-space:nowrap;display:inline-block"
    )
    if owner == "Manual":
        return (
            f'<span style="{chip_style};color:#888;background:#2a2e39">'
            "Manual</span>"
        )
    return (
        f'<span style="{chip_style};color:#0e1117;background:#4da6ff;'
        f'font-weight:600">🤖 {html.escape(owner)}</span>'
    )


_PF_POS_COLGROUP = (
    "<colgroup>"
    '<col style="width:13%"><col style="width:10%"><col style="width:9%">'
    '<col style="width:8%"><col style="width:7%"><col style="width:7%">'
    '<col style="width:9%"><col class="pf-col-bot" style="width:22%">'
    '<col style="width:12%">'
    "</colgroup>"
)


def _portfolio_position_row_cells(p: dict) -> str:
    symbol = html.escape((p.get("symbol") or "").strip() or str(p.get("instrument_id") or "—"))
    name = html.escape((p.get("name") or "").strip() or "—")
    direction = (p.get("direction") or "LONG").upper()
    dir_cls = "short" if direction == "SHORT" else "long"
    units = portfolio_units(p)
    units_txt = f"{units:,.4f}".rstrip("0").rstrip(".") if units else "—"
    net_val = p.get("current_value")
    if net_val is None and p.get("amount") is not None:
        net_val = float(p["amount"]) + float(p.get("pnl") or 0)
    pnl = p.get("pnl")
    pct = p.get("pnl_pct")
    pnl_col = ui.pnl_color(pnl)
    pnl_txt = fmt_money(pnl) if pnl is not None else "—"
    pct_txt = f"{pct:+.2f}%" if pct is not None else "—"
    live_tag = (
        f' <span style="color:{ui.C_LIVE};font-size:0.58rem">●</span>'
        if p.get("current_rate") is not None else ""
    )
    chg_html = portfolio_price_change_html(p)
    opened = html.escape(str(p.get("_open_display") or "—"))
    owner_html = _owner_chip_html(str(p.get("_owner") or "Manual"))
    return (
        f'<td class="pf-left"><p class="pf-symbol">{symbol}</p><p class="pf-name">{name}</p></td>'
        f'<td class="pf-right pf-col-price-gap">'
        f'<p class="pf-price">{fmt_market_price(p.get("current_rate"))}{live_tag}</p>'
        f"{chg_html}</td>"
        f'<td class="pf-left pf-col-units-pos"><p class="pf-units pf-units-line">'
        f"{units_txt}"
        f'<span class="pf-dir {dir_cls}">{direction.title()}</span></p></td>'
        f'<td class="pf-right"><p class="pf-val">{fmt_market_price(p.get("open_rate"))}</p></td>'
        f'<td class="pf-right"><p class="pf-pnl" style="color:{pnl_col}">{pnl_txt}</p></td>'
        f'<td class="pf-right"><p class="pf-pnl" style="color:{pnl_col}">{pct_txt}</p></td>'
        f'<td class="pf-right"><p class="pf-val">{"${:,.2f}".format(net_val) if net_val is not None else "—"}</p></td>'
        f'<td class="pf-left pf-col-bot">{owner_html}</td>'
        f'<td class="pf-right pf-col-opened"><p class="pf-val" style="font-size:0.82rem;color:#aaa">'
        f"{opened}</p></td>"
    )


def portfolio_positions_thead_html(n: int, *, with_close: bool = False) -> str:
    close_th = '<th class="pf-th pf-th-right" style="width:3%"></th>' if with_close else ""
    return (
        '<table class="pf-table pf-table-pos" style="margin-bottom:0">'
        f"{_PF_POS_COLGROUP}"
        "<thead><tr>"
        f'<th class="pf-th pf-th-left">Asset ({n})</th>'
        '<th class="pf-th pf-th-right pf-th-price-gap">Price</th>'
        '<th class="pf-th pf-th-left pf-th-units-pos">Units</th>'
        '<th class="pf-th pf-th-right">Avg. Open</th>'
        '<th class="pf-th pf-th-right">P/L</th>'
        '<th class="pf-th pf-th-right">P/L (%)</th>'
        '<th class="pf-th pf-th-right">Net Value</th>'
        '<th class="pf-th pf-th-left pf-th-bot">Bot</th>'
        '<th class="pf-th pf-th-right pf-th-opened">Opened</th>'
        f"{close_th}"
        "</tr></thead></table>"
    )


def portfolio_position_row_html(p: dict) -> str:
    """Single portfolio data row — pair with a Streamlit ✕ button beside it."""
    return (
        '<table class="pf-table pf-table-pos" style="margin:0">'
        f"{_PF_POS_COLGROUP}<tbody><tr>"
        f"{_portfolio_position_row_cells(p)}"
        "</tr></tbody></table>"
    )


def portfolio_positions_table_html(live_rows: list[dict]) -> str:
    """Full Portfolio-tab table — one HTML blob (bot + opened columns)."""
    rows_html = [f"<tr>{_portfolio_position_row_cells(p)}</tr>" for p in live_rows]
    n = len(live_rows)
    return (
        '<table class="pf-table pf-table-pos">'
        f"{_PF_POS_COLGROUP}"
        "<thead><tr>"
        f'<th class="pf-th pf-th-left">Asset ({n})</th>'
        '<th class="pf-th pf-th-right pf-th-price-gap">Price</th>'
        '<th class="pf-th pf-th-left pf-th-units-pos">Units</th>'
        '<th class="pf-th pf-th-right">Avg. Open</th>'
        '<th class="pf-th pf-th-right">P/L</th>'
        '<th class="pf-th pf-th-right">P/L (%)</th>'
        '<th class="pf-th pf-th-right">Net Value</th>'
        '<th class="pf-th pf-th-left pf-th-bot">Bot</th>'
        '<th class="pf-th pf-th-right pf-th-opened">Opened</th>'
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def history_trades_table_html(trades: list[dict]) -> str:
    rows_html = []
    for t in trades:
        iid = t.get("instrumentId") or t.get("instrument_id")
        try:
            iid_int = int(iid) if iid is not None else None
        except (TypeError, ValueError):
            iid_int = None

        is_buy = t.get("isBuy")
        stock = html.escape(display_asset_name(_label_for_id(iid_int)))
        side = "Bought" if is_buy is True else ("Sold" if is_buy is False else "—")
        direction = "LONG" if is_buy is True else ("SHORT" if is_buy is False else "—")
        dir_cls = "short" if direction == "SHORT" else "long"

        open_rate = t.get("openRate")
        close_rate = t.get("closeRate")
        pnl = t.get("netProfit")
        invested = t.get("investment")
        units = t.get("units")

        pnl_col = ui.pnl_color(float(pnl) if pnl is not None else None)
        pnl_txt = f"${float(pnl):+,.2f}" if pnl is not None else "—"
        open_txt = f"{float(open_rate):.5f}" if open_rate is not None else "—"
        close_txt = f"{float(close_rate):.5f}" if close_rate is not None else "—"
        inv_txt = f"${float(invested):,.2f}" if invested is not None else "—"
        units_txt = (
            f"{float(units):,.4f}".rstrip("0").rstrip(".")
            if units is not None else "—"
        )

        owner_html = _owner_chip_html(str(t.get("_owner") or "Manual"))

        close_method_html = close_method_badge_html(
            t.get("_close_reason", ""),
            t.get("_close_strategy", ""),
        )

        rows_html.append(
            f"<tr>"
            f'<td class="pf-left"><p class="pf-symbol">{stock}</p>'
            f'<p class="pf-name"><span class="pf-dir {dir_cls}">{html.escape(side)}</span>'
            f" · {html.escape(direction)}</p></td>"
            f'<td class="pf-left pf-col-bot">{owner_html}</td>'
            f'<td class="pf-left">{close_method_html}</td>'
            f'<td class="pf-right"><p class="pf-val">{open_txt}</p></td>'
            f'<td class="pf-right"><p class="pf-val">{close_txt}</p></td>'
            f'<td class="pf-right"><p class="pf-pnl" style="color:{pnl_col}">{pnl_txt}</p></td>'
            f'<td class="pf-right"><p class="pf-val">{inv_txt}</p></td>'
            f'<td class="pf-right pf-col-units-gap"><p class="pf-units">{units_txt}</p></td>'
            f'<td class="pf-left pf-col-opened"><p class="pf-ts">'
            f'{html.escape(parse_api_timestamp_short(t.get("openTimestamp")))}</p></td>'
            f'<td class="pf-left pf-col-closed"><p class="pf-ts">'
            f'{html.escape(parse_api_timestamp_short(t.get("closeTimestamp")))}</p></td>'
            f"</tr>"
        )

    n = len(trades)
    return (
        '<table class="pf-table pf-table-hist">'
        "<colgroup>"
        '<col style="width:12%"><col class="pf-col-bot" style="width:22%">'
        '<col style="width:11%"><col style="width:7%"><col style="width:7%">'
        '<col style="width:7%"><col style="width:8%"><col style="width:6%">'
        '<col style="width:10%"><col style="width:10%">'
        "</colgroup>"
        "<thead><tr>"
        f'<th class="pf-th pf-th-left">Stock ({n})</th>'
        '<th class="pf-th pf-th-left pf-th-bot">Bot</th>'
        '<th class="pf-th pf-th-left">Close method</th>'
        '<th class="pf-th pf-th-right">Open @</th>'
        '<th class="pf-th pf-th-right">Close @</th>'
        '<th class="pf-th pf-th-right">P/L</th>'
        '<th class="pf-th pf-th-right">Invested</th>'
        '<th class="pf-th pf-th-right pf-th-units-gap">Units</th>'
        '<th class="pf-th pf-th-left pf-th-opened">Opened</th>'
        '<th class="pf-th pf-th-left pf-th-closed">Closed</th>'
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def closed_trades_block_html(trades: list[dict]) -> str:
    pnl_vals = [
        float(t["netProfit"])
        for t in trades
        if t.get("netProfit") is not None
    ]
    total = sum(pnl_vals) if pnl_vals else 0.0
    wins = sum(1 for v in pnl_vals if v > 0)
    losses = sum(1 for v in pnl_vals if v < 0)
    wr = wins / len(pnl_vals) * 100 if pnl_vals else 0
    stats = history_stats_html([
        ("Closed trades", str(len(trades)), None),
        ("Realised P&L", f"${total:,.2f}", ui.pnl_color(total)),
        ("Win rate", f"{wr:.1f}%", None),
        ("W / L", f"{wins} / {losses}", None),
    ])
    return (
        f'<p class="pf-section-title">Closed trades</p>'
        f"{stats}"
        f"{history_trades_table_html(trades)}"
    )


def open_positions_history_html(live_rows: list[dict]) -> str:
    n = len(live_rows)
    total_pnl = sum(r.get("pnl") or 0 for r in live_rows)
    summary = history_stats_html([
        ("Open", str(n), None),
        ("Unrealised P&L", fmt_money(total_pnl), ui.pnl_color(total_pnl)),
    ])
    if n == 0:
        return summary
    return summary + portfolio_table_html(live_rows)
