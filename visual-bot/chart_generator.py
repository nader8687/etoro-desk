"""
Renders a candlestick chart to PNG bytes using mplfinance.
No browser / kaleido dependency — works reliably in any Docker container.
"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplfinance as mpf


_STYLE = mpf.make_mpf_style(
    base_mpl_style="dark_background",
    marketcolors=mpf.make_marketcolors(
        up="#089981", down="#f23645",
        edge="inherit", wick="inherit",
        volume={"up": "#089981", "down": "#f23645"},
    ),
    gridstyle="--",
    gridcolor="#1e222d",
    facecolor="#131722",
    edgecolor="#131722",
    figcolor="#131722",
    rc={"axes.labelcolor": "#787b86", "xtick.color": "#787b86",
        "ytick.color": "#787b86", "axes.edgecolor": "#2a2e39"},
)


def candles_to_image(
    df: pd.DataFrame,
    title: str = "Chart",
    add_ema20: bool = True,
    add_ema50: bool = True,
    *,
    entry_price: float | None = None,
    direction: str | None = None,   # "LONG" or "SHORT"
    minutes_open: float = 0.0,      # used to locate entry candle on chart
) -> bytes:
    """
    df must have columns: time (tz-aware), Open, High, Low, Close, [Volume].
    Returns PNG bytes.

    If entry_price + direction are provided, an entry arrow and a horizontal
    dashed line are drawn on the chart so the LLM can see exactly where the
    position was opened.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — no candles to render.")

    plot_df = df.copy()
    plot_df["time"] = pd.to_datetime(plot_df["time"]).dt.tz_convert(None)
    plot_df = plot_df.set_index("time")
    plot_df.index.name = "Date"

    # Always drop Volume if not usable — mplfinance detects it from the column
    if "Volume" in plot_df.columns:
        vol = pd.to_numeric(plot_df["Volume"], errors="coerce").fillna(0)
        if vol.sum() > 0:
            plot_df["Volume"] = vol
            has_vol = True
        else:
            plot_df = plot_df.drop(columns=["Volume"])
            has_vol = False
    else:
        has_vol = False

    apds = []
    if add_ema20 and len(plot_df) >= 20:
        ema20 = plot_df["Close"].ewm(span=20, adjust=False).mean()
        apds.append(mpf.make_addplot(ema20, color="#f0c040", width=1.2, label="EMA 20"))

    if add_ema50 and len(plot_df) >= 50:
        ema50 = plot_df["Close"].ewm(span=50, adjust=False).mean()
        apds.append(mpf.make_addplot(ema50, color="#4da6ff", width=1.2, label="EMA 50"))

    buf = io.BytesIO()
    plot_kwargs = dict(
        type="candle",
        style=_STYLE,
        title=dict(title=f"\n{title}", color="#d1d4dc", size=13),
        addplot=apds if apds else None,
        figsize=(10, 6),
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=9999,
        datetime_format="%H:%M" if len(plot_df) <= 300 else "%m/%d",
    )
    if has_vol:
        plot_kwargs["volume"] = True   # only pass when True — never pass False

    fig, axes = mpf.plot(plot_df, **plot_kwargs)

    # Price tag on last candle
    ax_price = axes[0]
    last_price = float(plot_df["Close"].iloc[-1])
    color = "#089981" if plot_df["Close"].iloc[-1] >= plot_df["Open"].iloc[-1] else "#f23645"
    ax_price.axhline(last_price, color=color, linewidth=0.8, linestyle=":")
    ax_price.annotate(
        f" {last_price:.5f} ",
        xy=(1, last_price), xycoords=("axes fraction", "data"),
        ha="left", va="center",
        fontsize=9, color="#131722",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor=color),
    )

    # ── Entry price marker ────────────────────────────────────────────────────
    if entry_price is not None and direction is not None:
        dir_upper   = direction.upper()
        entry_color = "#089981" if dir_upper == "LONG" else "#f23645"
        dir_label   = "▲ LONG" if dir_upper == "LONG" else "▼ SHORT"
        n           = len(plot_df)

        # Horizontal dashed line at entry price across the full chart
        ax_price.axhline(
            entry_price,
            color=entry_color, linewidth=1.2, linestyle="--", alpha=0.88, zorder=3,
        )

        # Label on the right axis (same style as the current-price tag)
        ax_price.annotate(
            f" {dir_label}  {entry_price:.5f} ",
            xy=(1.0, entry_price), xycoords=("axes fraction", "data"),
            ha="left", va="center",
            fontsize=8, color="#131722",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=entry_color, edgecolor=entry_color),
            zorder=5,
        )

        # ── Entry-candle arrow ────────────────────────────────────────────────
        # Estimate which candle the trade was opened on from minutes_open.
        # mplfinance uses integer x-axis (0 … n-1), so we work in those units.
        if minutes_open > 0 and n >= 2:
            interval_secs = max(
                (plot_df.index[-1] - plot_df.index[-2]).total_seconds(), 1
            )
            candles_ago = int(round(minutes_open * 60 / interval_secs))
            entry_idx = max(0, n - 1 - candles_ago)
        else:
            entry_idx = 0   # oldest visible candle when age unknown

        price_range = max(float(plot_df["High"].max() - plot_df["Low"].min()), entry_price * 0.001)
        offset      = price_range * 0.03   # 3% of visible range — tail distance

        if dir_upper == "LONG":
            tail_y = entry_price - offset   # tail below the entry line
        else:
            tail_y = entry_price + offset   # tail above the entry line

        ax_price.annotate(
            "",
            xy=(entry_idx, entry_price),   # arrowhead sits on the entry price line
            xytext=(entry_idx, tail_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=entry_color,
                lw=2.0,
                mutation_scale=14,
            ),
            zorder=6,
        )

    if apds:
        ax_price.legend(
            handles=[
                plt.Line2D([0], [0], color="#f0c040", linewidth=1.2, label="EMA 20"),
                plt.Line2D([0], [0], color="#4da6ff", linewidth=1.2, label="EMA 50"),
            ],
            loc="upper left", fontsize=8,
            facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="#d1d4dc",
        )

    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#131722", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
