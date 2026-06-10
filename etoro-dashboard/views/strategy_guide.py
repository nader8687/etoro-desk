"""Strategy Guide — interactive reference for every trading strategy in EtoroDesk."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── palette (matches app dark theme) ─────────────────────────────────────────
C_BG      = "#0e1117"
C_GRID    = "#1e2130"
C_TEXT    = "#e0e0e0"
C_MUTED   = "#8888aa"
C_GREEN   = "#00c896"
C_RED     = "#ff4d6d"
C_BLUE    = "#4d9fff"
C_YELLOW  = "#ffd166"
C_PURPLE  = "#c084fc"
C_ORANGE  = "#fb923c"
C_TEAL    = "#22d3ee"

_PLOTLY_CFG = dict(displayModeBar=False, staticPlot=False)


def _fig_base(rows: int = 1, cols: int = 1, row_heights=None, shared_xaxes=True) -> go.Figure:
    fig = make_subplots(
        rows=rows, cols=cols,
        shared_xaxes=shared_xaxes,
        row_heights=row_heights or ([1] * rows),
        vertical_spacing=0.06,
    )
    fig.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        font_color=C_TEXT, font_size=11,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font_size=10,
            orientation="h", x=0, y=1.05,
        ),
        height=340,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(gridcolor=C_GRID, zeroline=False),
    )
    for i in range(2, rows + 1):
        fig.update_layout(**{
            f"xaxis{i}": dict(showgrid=False, zeroline=False, showticklabels=False),
            f"yaxis{i}": dict(gridcolor=C_GRID, zeroline=False),
        })
    return fig


def _candles(n: int = 120, seed: int = 42, trend: float = 0.0002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = [1.0000]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.normal(0, 0.003)))
    closes = np.array(closes)
    high   = closes * (1 + np.abs(rng.normal(0, 0.002, n)))
    low    = closes * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_  = np.roll(closes, 1)
    open_[0] = closes[0]
    volume = rng.integers(1000, 5000, n).astype(float)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": closes, "Volume": volume})
    return df


def _add_candle_trace(fig, df, row=1, col=1, name="Price"):
    fig.add_trace(go.Candlestick(
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
        name=name, showlegend=False,
        increasing_fillcolor=C_GREEN, decreasing_fillcolor=C_RED,
    ), row=row, col=col)
    fig.update_traces(selector=dict(type="candlestick"), line_width=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy data classes
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_META: dict[str, dict] = {

    "llm": {
        "display_name": "AI Vision (LLM)",
        "category": "AI",
        "icon": "🤖",
        "tagline": "A vision-language model reads the chart as an image and decides like a human trader.",
        "how_it_works": [
            "Every candle close, the live OHLC chart is rendered as a **PNG image**.",
            "The image is sent to a GPT-4o Vision model with a structured prompt describing the asset, timeframe, spread, and current position.",
            "The model analyses: trend direction, momentum, support/resistance, rejection candles, reversal risk, and whether the expected move covers the spread.",
            "It returns structured JSON: signal (BUY_LONG / SELL_SHORT / HOLD_NO_POSITION), confidence (0–100), reasoning, nearest support/resistance.",
            "An **execution quality gate** then filters out signals where net edge ≤ 0 after slippage — only high-conviction, positive-EV trades are executed.",
        ],
        "signals": {
            "BUY":  "LLM sees a bullish setup with upside clearly larger than spread/fees and momentum supporting continuation.",
            "SELL": "LLM sees a bearish setup with downside clearly larger than spread/fees.",
            "HOLD": "Setup is weak, late, unclear, or too close to S/R. LLM holds rather than forcing a trade.",
        },
        "parameters": [
            ("Model", "GPT-4o-mini (configurable)", "Vision LLM used for chart analysis"),
            ("Candles shown", "300 (configurable)", "Chart window fed to the model"),
            ("Confidence threshold", "Execution quality gate", "Only viable signals (net_edge > 0) are traded"),
        ],
        "pros": [
            "Context-aware: considers support, resistance, trend, candlestick patterns simultaneously",
            "Adaptable — no fixed rules, adapts to different market conditions",
            "Explains its reasoning in plain English",
        ],
        "cons": [
            "Slower than rule-based strategies (3–5 seconds per call)",
            "Costs per API call — not free at scale",
            "Non-deterministic: same chart may produce slightly different signals",
            "Quota dependent — stops working if OpenAI limit is hit",
        ],
        "best_for": "Any asset and timeframe. Most powerful on 5m+ candles where chart patterns have more meaning.",
    },

    "supertrend": {
        "display_name": "Supertrend (10, 3)",
        "category": "Trend-Following",
        "icon": "📈",
        "tagline": "ATR-based trailing stop line that flips sides on trend reversals.",
        "how_it_works": [
            "Computes the **Average True Range (ATR)** over 10 periods using Wilder smoothing.",
            "Draws two bands: `HL2 ± 3×ATR` where HL2 = (High + Low) / 2.",
            "The **upper band** ratchets down (never goes higher) while price is below it.",
            "The **lower band** ratchets up (never goes lower) while price is above it.",
            "**Direction flips bullish** when Close crosses above the upper band → BUY signal.",
            "**Direction flips bearish** when Close crosses below the lower band → SELL signal.",
            "Silent (HOLD) on every candle where the trend is sustained without flipping.",
        ],
        "signals": {
            "BUY":  "Trend flipped from bearish to bullish this candle.",
            "SELL": "Trend flipped from bullish to bearish this candle.",
            "HOLD": "Trend direction unchanged — no flip this candle.",
        },
        "parameters": [
            ("ATR Period", "10", "Smoothing window for average true range"),
            ("Multiplier", "3.0", "Band width = multiplier × ATR"),
        ],
        "pros": [
            "Low noise — only fires on trend flips, not every candle",
            "Self-adjusting: wider bands in volatile markets (ATR expands)",
            "Clear, unambiguous entry/exit rules",
        ],
        "cons": [
            "Lagging — the flip happens after the trend change, not before",
            "Whipsaws badly in choppy/ranging markets",
            "Fixed confidence (78) — doesn't scale with flip strength",
        ],
        "best_for": "Trending markets. BTC/XRP during strong directional moves. Works best on 5m–15m candles.",
    },

    "macd": {
        "display_name": "MACD (12/26/9)",
        "category": "Trend-Following",
        "icon": "〰️",
        "tagline": "Measures momentum acceleration via crossovers of two exponential moving averages.",
        "how_it_works": [
            "**Fast EMA (12)** tracks recent price closely.",
            "**Slow EMA (26)** is smoother and lags more.",
            "**MACD line** = Fast EMA − Slow EMA. Positive = upward momentum.",
            "**Signal line (9)** = 9-period EMA of the MACD line.",
            "**Histogram** = MACD − Signal. Growing histogram = accelerating momentum.",
            "**BUY crossover**: MACD line crosses above Signal line.",
            "**SELL crossover**: MACD line crosses below Signal line.",
            "Confidence scales with histogram magnitude — a strong histogram at crossover = higher confidence.",
        ],
        "signals": {
            "BUY":  "MACD line crossed above the signal line this candle.",
            "SELL": "MACD line crossed below the signal line this candle.",
            "HOLD": "No fresh crossover — MACD is trending but not flipping.",
        },
        "parameters": [
            ("Fast EMA", "12", "Short-term momentum EMA"),
            ("Slow EMA", "26", "Long-term trend EMA"),
            ("Signal", "9",  "EMA of the MACD line for crossovers"),
        ],
        "pros": [
            "Captures both trend direction and momentum acceleration",
            "Histogram gives early warning before the crossover",
            "Well-tested across decades of market data",
        ],
        "cons": [
            "Lagging — EMAs react after price moves",
            "Many false signals in sideways markets",
            "Needs 38+ candles to initialise correctly",
        ],
        "best_for": "Trending crypto markets. Combine with RSI to avoid overbought entries.",
    },

    "rsi": {
        "display_name": "RSI (14)",
        "category": "Oscillator",
        "icon": "🔄",
        "tagline": "Measures overbought and oversold conditions using the ratio of average gains to losses.",
        "how_it_works": [
            "Calculates average gains and average losses over 14 candles using EWM smoothing.",
            "**RSI = 100 − (100 / (1 + RS))** where RS = avg_gain / avg_loss.",
            "RSI oscillates between 0 and 100.",
            "**Oversold (≤ 30)**: price has fallen too fast, mean-reversion bounce likely → BUY.",
            "**Overbought (≥ 70)**: price has risen too fast, pullback likely → SELL.",
            "Confidence increases the deeper RSI goes past the threshold (RSI 10 = higher confidence than RSI 28).",
        ],
        "signals": {
            "BUY":  "RSI ≤ 30 — oversold. Expected bounce back toward the mean.",
            "SELL": "RSI ≥ 70 — overbought. Expected pullback.",
            "HOLD": "RSI between 30–70 — neutral zone, no clear extreme.",
        },
        "parameters": [
            ("Period", "14", "Lookback window for gain/loss averaging"),
            ("Oversold", "30", "RSI level that triggers BUY"),
            ("Overbought", "70", "RSI level that triggers SELL"),
        ],
        "pros": [
            "Excellent for identifying turning points in ranging markets",
            "Simple and highly interpretable",
            "Scales confidence naturally with RSI extreme depth",
        ],
        "cons": [
            "In strong trends, RSI stays overbought for a long time (false SELL signals)",
            "Not suitable as a standalone trend-following tool",
            "Crosses happen frequently in volatile markets",
        ],
        "best_for": "Ranging/sideways markets. Use with MACD to filter trend direction before taking RSI signals.",
    },

    "ma_crossover": {
        "display_name": "MA Crossover (9/21 EMA)",
        "category": "Trend-Following",
        "icon": "✂️",
        "tagline": "Two exponential moving averages — the 'golden cross' and 'death cross' system.",
        "how_it_works": [
            "**Fast EMA (9)** reacts quickly to price changes.",
            "**Slow EMA (21)** is smoother and represents the medium-term trend.",
            "**Golden cross**: Fast EMA crosses above Slow EMA → BUY (bullish trend starting).",
            "**Death cross**: Fast EMA crosses below Slow EMA → SELL (bearish trend starting).",
            "Confidence scales with the gap between the two EMAs at crossover — wider gap = stronger signal.",
        ],
        "signals": {
            "BUY":  "EMA 9 crossed above EMA 21 — bullish trend starting.",
            "SELL": "EMA 9 crossed below EMA 21 — bearish trend starting.",
            "HOLD": "No crossover this candle. EMAs trending but not flipping.",
        },
        "parameters": [
            ("Fast EMA", "9",  "Short-term momentum average"),
            ("Slow EMA", "21", "Medium-term trend average"),
        ],
        "pros": [
            "One of the most time-tested strategies in technical analysis",
            "Smooths out noise better than price crossovers",
            "Works well across all timeframes",
        ],
        "cons": [
            "Lagging — crossover confirms trend change after it happens",
            "Frequent whipsaws during consolidation",
        ],
        "best_for": "Medium-term crypto trends. Best on 15m+ candles.",
    },

    "adx": {
        "display_name": "ADX Trend Strength (14)",
        "category": "Trend-Following",
        "icon": "💪",
        "tagline": "Measures how STRONG a trend is — not just direction, but conviction.",
        "how_it_works": [
            "**+DM / −DM**: Directional movement — how much price made a new high vs new low vs prior candle.",
            "**+DI / −DI**: Wilder-smoothed versions of +DM and −DM, expressed as % of ATR.",
            "**DX**: measures the divergence between +DI and −DI: `100 × |+DI − −DI| / (+DI + −DI)`.",
            "**ADX**: Wilder-smoothed DX. ADX ≥ 25 = trending market. ADX < 20 = no trend.",
            "**BUY**: +DI crosses above −DI AND ADX ≥ 25 (strong bullish trend confirmed).",
            "**SELL**: −DI crosses above +DI AND ADX ≥ 25 (strong bearish trend confirmed).",
            "If ADX < 25: HOLD regardless of crossover — trend is too weak to trade.",
        ],
        "signals": {
            "BUY":  "+DI crossed above −DI with ADX ≥ 25 — strong bullish directional move.",
            "SELL": "−DI crossed above +DI with ADX ≥ 25 — strong bearish directional move.",
            "HOLD": "No crossover or ADX < 25 — trending but not strong enough, or ranging.",
        },
        "parameters": [
            ("Period", "14", "Wilder smoothing window for DI and ADX"),
            ("ADX Threshold", "25", "Minimum ADX to confirm a valid trend"),
        ],
        "pros": [
            "Unique: explicitly measures trend STRENGTH, not just direction",
            "Great filter — avoids false signals in ranging markets",
            "Can be combined with other strategies as a 'trend qualifier'",
        ],
        "cons": [
            "Very selective — rarely fires in non-trending conditions",
            "ADX lags price significantly",
        ],
        "best_for": "Strong trending markets. Excellent as a secondary filter alongside MACD or MA Crossover.",
    },

    "ichimoku": {
        "display_name": "Ichimoku Cloud",
        "category": "Trend-Following",
        "icon": "☁️",
        "tagline": "A complete self-contained trading system using five components and a scoring system.",
        "how_it_works": [
            "**Tenkan-sen (9)**: (9-period high + low) / 2 — the 'conversion line', short-term trend.",
            "**Kijun-sen (26)**: (26-period high + low) / 2 — the 'base line', medium-term trend.",
            "**Senkou Span A**: (Tenkan + Kijun) / 2, projected 26 bars forward — cloud boundary 1.",
            "**Senkou Span B (52)**: (52-period high + low) / 2, projected 26 bars forward — cloud boundary 2.",
            "**Chikou Span**: Current close plotted 26 bars back — confirms trend from the past.",
            "Four components scored: Price vs cloud (+25), Cloud colour (+20), TK cross (+20), Chikou (+15).",
            "**BUY** when total score ≥ 55. **SELL** when ≤ −55. Otherwise HOLD.",
        ],
        "signals": {
            "BUY":  "Score ≥ 55: price above cloud + bullish cloud + Tenkan > Kijun + Chikou confirms.",
            "SELL": "Score ≤ −55: price below cloud + bearish cloud + Tenkan < Kijun + Chikou confirms.",
            "HOLD": "Mixed signals — not enough components aligned for high confidence.",
        },
        "parameters": [
            ("Tenkan", "9",  "Conversion line period"),
            ("Kijun",  "26", "Base line period"),
            ("Senkou B", "52", "Slow cloud boundary period"),
            ("Displacement", "26", "Forward projection offset"),
            ("Buy threshold",  "55", "Minimum score for BUY"),
            ("Sell threshold", "−55", "Maximum score for SELL"),
        ],
        "pros": [
            "Multi-dimensional: trend, momentum, support/resistance in one system",
            "Cloud acts as dynamic support and resistance zone",
            "Self-confirming: requires multiple aligned signals",
        ],
        "cons": [
            "Needs 78+ candles to initialise (52 + 26 displacement)",
            "Complex to interpret manually",
            "Designed for daily charts — weaker on 1-minute",
        ],
        "best_for": "Higher timeframes (1h, 4h, daily). On 1-minute, use as a direction filter only.",
    },

    "bollinger_squeeze": {
        "display_name": "Bollinger Band Squeeze",
        "category": "Breakout",
        "icon": "🎯",
        "tagline": "Detects low-volatility compressions and trades the explosive breakout that follows.",
        "how_it_works": [
            "**Bollinger Bands**: 20-period SMA ± 2 standard deviations.",
            "**BB Width** = (Upper − Lower) / Middle. Narrow width = low volatility (squeeze).",
            "Squeeze is active when current BB width ≤ 20th percentile of recent widths — statistically tight.",
            "**BUY**: price closes above the upper band WHILE in a squeeze — volatility breakout upward.",
            "**SELL**: price closes below the lower band WHILE in a squeeze — volatility breakout downward.",
            "The squeeze itself is silent (HOLD) — waiting for the direction to reveal itself.",
        ],
        "signals": {
            "BUY":  "Price broke above the upper Bollinger Band during a volatility squeeze.",
            "SELL": "Price broke below the lower Bollinger Band during a volatility squeeze.",
            "HOLD": "In a squeeze but no breakout yet, or no squeeze active.",
        },
        "parameters": [
            ("Period", "20", "SMA and standard deviation window"),
            ("Std dev", "2.0", "Band width multiplier"),
            ("Squeeze pct", "20th pct", "Percentile threshold for squeeze detection"),
        ],
        "pros": [
            "Excellent at catching explosive moves after consolidation",
            "Percentile-based detection is robust to different asset volatility levels",
            "Very high conviction when it fires — compression + breakout together",
        ],
        "cons": [
            "May miss breakouts that start without a squeeze",
            "Breakout direction can fail (false breakouts)",
            "Can be silent for many candles during normal volatility",
        ],
        "best_for": "Crypto assets that consolidate then explode. BTC before major news/events.",
    },

    "donchian": {
        "display_name": "Donchian Channel Breakout (20)",
        "category": "Breakout",
        "icon": "📦",
        "tagline": "Trades breakouts above or below the 20-period price range.",
        "how_it_works": [
            "**Upper band**: 20-period rolling maximum of High prices.",
            "**Lower band**: 20-period rolling minimum of Low prices.",
            "**BUY**: current Close exceeds the upper band (new 20-period high).",
            "**SELL**: current Close falls below the lower band (new 20-period low).",
            "Confidence scales with how far beyond the band the price has moved.",
        ],
        "signals": {
            "BUY":  "Price closed above the 20-period high — new breakout upward.",
            "SELL": "Price closed below the 20-period low — new breakdown downward.",
            "HOLD": "Price within the channel — no breakout this candle.",
        },
        "parameters": [
            ("Period", "20", "Rolling high/low window"),
        ],
        "pros": [
            "Simple, clean logic — new highs/lows are objectively meaningful",
            "Turtles Trading System was built on Donchian breakouts",
            "Works well in strongly trending crypto markets",
        ],
        "cons": [
            "Entries are at new highs/lows — feels counter-intuitive",
            "Prone to false breakouts in choppy markets",
        ],
        "best_for": "Trending crypto markets. Combine with ADX to confirm trend strength before entry.",
    },

    "stoch_rsi": {
        "display_name": "Stochastic RSI",
        "category": "Oscillator",
        "icon": "🌊",
        "tagline": "Applies the Stochastic formula to RSI values for faster, more sensitive signals.",
        "how_it_works": [
            "**RSI (14)** is computed first using standard Wilder EWM smoothing.",
            "**Stochastic RSI** = (RSI − min RSI over 14) / (max RSI − min RSI over 14). Range: 0–1.",
            "**%K**: smoothed StochRSI (3-period SMA). **%D**: 3-period SMA of %K (signal line).",
            "**BUY**: %K crosses above %D from below 0.20 (oversold zone).",
            "**SELL**: %K crosses below %D from above 0.80 (overbought zone).",
            "Fires earlier than plain RSI — useful for fast-moving crypto.",
        ],
        "signals": {
            "BUY":  "%K crossed above %D in the oversold zone (< 0.20).",
            "SELL": "%K crossed below %D in the overbought zone (> 0.80).",
            "HOLD": "No crossover in extreme zone, or oscillator in neutral range.",
        },
        "parameters": [
            ("RSI Period", "14", "Base RSI lookback"),
            ("Stoch Period", "14", "Stochastic lookback on the RSI values"),
            ("K Period", "3",  "Smoothing for %K line"),
            ("D Period", "3",  "Signal line smoothing"),
        ],
        "pros": [
            "More sensitive than RSI — reacts faster to price changes",
            "Double confirmation: both stochastic position AND crossover required",
            "Good for scalping on 1m–5m candles",
        ],
        "cons": [
            "More false signals than RSI due to higher sensitivity",
            "Requires ~35 candles to initialise",
        ],
        "best_for": "Fast-moving 1m–5m crypto scalping. Use with a trend filter to avoid counter-trend trades.",
    },

    "mean_reversion": {
        "display_name": "Mean Reversion Z-score (20)",
        "category": "Oscillator",
        "icon": "⚖️",
        "tagline": "Measures how many standard deviations price is from its recent mean — then fades the extreme.",
        "how_it_works": [
            "Calculates the **20-period rolling mean** and **standard deviation** of Close prices.",
            "**Z-score** = (Close − mean) / std. A Z-score of +2 means price is 2 std devs above average.",
            "**SELL (fade)** when Z > +2.0: price is statistically stretched above the mean.",
            "**BUY (fade)** when Z < −2.0: price is statistically stretched below the mean.",
            "Confidence = min(50 + |Z| × 20, 92) — deeper extremes = higher confidence.",
        ],
        "signals": {
            "BUY":  "Z-score < −2.0 — price significantly below recent mean, bounce expected.",
            "SELL": "Z-score > +2.0 — price significantly above recent mean, pullback expected.",
            "HOLD": "Z-score within ±2.0 — within normal statistical range.",
        },
        "parameters": [
            ("Period", "20", "Rolling window for mean and std dev"),
            ("Buy threshold",  "−2.0", "Z-score below this triggers BUY"),
            ("Sell threshold", "+2.0", "Z-score above this triggers SELL"),
        ],
        "pros": [
            "Statistically grounded — 2 std dev events are genuinely rare",
            "Scales confidence with extremity of the move",
            "Works well in low-volatility FX pairs that revert",
        ],
        "cons": [
            "Catastrophic in strong trends — price can stay above +2 for many candles",
            "Mean itself shifts during trends — baseline is moving",
        ],
        "best_for": "Range-bound FX pairs. Dangerous on trending crypto without a trend filter.",
    },

    "candlestick": {
        "display_name": "Candlestick Patterns",
        "category": "Price Action",
        "icon": "🕯️",
        "tagline": "Pure price action — detects 12 named candlestick patterns and scores their combined signal.",
        "how_it_works": [
            "Scans the **last 3 candles** for named bullish and bearish patterns.",
            "Each pattern has a fixed score: **+40** (Three White Soldiers) down to **±5** (Doji).",
            "Bullish patterns add to the score; bearish patterns subtract.",
            "**BUY** when net score ≥ +20. **SELL** when ≤ −20.",
            "Confidence = min(50 + |net_score|, 92).",
            "Patterns detected: Engulfing, Hammer, Shooting Star, Pin Bar, Morning/Evening Star, Three Soldiers/Crows, Doji.",
        ],
        "signals": {
            "BUY":  "Net bullish pattern score ≥ 20 (e.g. Bullish Engulfing or Three White Soldiers).",
            "SELL": "Net bearish pattern score ≤ −20 (e.g. Bearish Engulfing or Three Black Crows).",
            "HOLD": "No significant pattern or mixed bullish/bearish patterns cancel out.",
        },
        "parameters": [
            ("Score threshold", "±20", "Minimum absolute score to fire a signal"),
        ],
        "pros": [
            "Zero indicators — purely reacts to what price IS doing right now",
            "Fast: only needs last 3 candles",
            "Readable: you can visually verify the pattern on the chart",
        ],
        "cons": [
            "Pattern recognition is mechanical — context-blind",
            "Many candles that 'look like' patterns are noise at 1-minute timeframe",
            "Best results on daily/4h candles where patterns carry more weight",
        ],
        "best_for": "Daily and 4h charts. At 1-minute, combine with trend filter to avoid low-quality signals.",
    },

    "orb": {
        "display_name": "Opening Range Breakout (ORB)",
        "category": "Breakout",
        "icon": "🚀",
        "tagline": "Trades breakouts above or below the price range set in the first N candles of a session.",
        "how_it_works": [
            "Defines an **opening range** using the first 5 candles of available data.",
            "**Opening range high** = max High of those candles.",
            "**Opening range low** = min Low of those candles.",
            "**BUY**: price closes above the opening range high — bullish breakout.",
            "**SELL**: price closes below the opening range low — bearish breakdown.",
            "Designed for markets with a distinct session open (equities, futures).",
        ],
        "signals": {
            "BUY":  "Price broke above the session opening range high.",
            "SELL": "Price broke below the session opening range low.",
            "HOLD": "Price still within the opening range.",
        },
        "parameters": [
            ("Range candles", "5", "Number of candles defining the opening range"),
        ],
        "pros": [
            "Captures the directional bias set at market open",
            "Simple and objective",
        ],
        "cons": [
            "Crypto trades 24/7 — there is no formal 'open', so the range is the first N candles loaded",
            "Range can reset on restart — not persistent across sessions",
            "Less meaningful for 24/7 assets like BTC/XRP",
        ],
        "best_for": "Traditional equities and futures with a defined market open. Less suitable for 24/7 crypto.",
    },

    "stat_arb": {
        "display_name": "Statistical Arbitrage (Pairs)",
        "category": "Arbitrage",
        "icon": "🔗",
        "tagline": "Exploits temporary price divergences between two historically correlated assets.",
        "how_it_works": [
            "Designed for **two correlated instruments** (e.g. BTC vs ETH).",
            "Tracks the **spread** = price of asset A − hedge_ratio × price of asset B.",
            "Computes a **Z-score** of the spread vs its rolling mean and standard deviation.",
            "**BUY A** when spread Z-score < −2 (A cheap vs B).",
            "**SELL A** when spread Z-score > +2 (A expensive vs B).",
            "Requires both assets to be trading concurrently — single-instrument fallback uses autocorrelation.",
        ],
        "signals": {
            "BUY":  "Spread significantly below mean — asset A is undervalued relative to its pair.",
            "SELL": "Spread significantly above mean — asset A is overvalued relative to its pair.",
            "HOLD": "Spread within normal range — no arbitrage opportunity.",
        },
        "parameters": [
            ("Period", "30", "Rolling window for spread mean/std"),
            ("Entry Z", "±2.0", "Z-score threshold for entry"),
        ],
        "pros": [
            "Market-neutral: profits from relative mispricing, not absolute direction",
            "Theoretically lower risk than directional trading",
        ],
        "cons": [
            "Requires two correlated instruments — hard to run single-instrument",
            "Correlation can break down (regime change)",
            "Complex to manage open positions on both legs",
        ],
        "best_for": "Running two correlated crypto pairs simultaneously (e.g. BTC + ETH bots both active).",
    },

    "rate_arb": {
        "display_name": "Rate Arbitrage",
        "category": "Arbitrage",
        "icon": "💱",
        "tagline": "Exploits momentum signals derived from cross-asset rate differentials.",
        "how_it_works": [
            "Monitors the **rate-of-change (ROC)** of the current instrument over 5 and 20 periods.",
            "Computes a **momentum ratio** = short ROC / long ROC to detect acceleration.",
            "**BUY** when short momentum > long momentum with significant positive differential.",
            "**SELL** when short momentum < long momentum with significant negative differential.",
            "Primarily a momentum strategy using multi-timeframe ROC as a proxy for rate arbitrage.",
        ],
        "signals": {
            "BUY":  "Short-term rate-of-change significantly exceeds long-term — accelerating upward.",
            "SELL": "Short-term rate-of-change significantly below long-term — accelerating downward.",
            "HOLD": "No significant rate differential between timeframes.",
        },
        "parameters": [
            ("Short period", "5",  "Short-term ROC window"),
            ("Long period",  "20", "Long-term ROC window"),
        ],
        "pros": [
            "Multi-timeframe momentum perspective",
            "Can catch early trend acceleration",
        ],
        "cons": [
            "Experimental — not a traditional arbitrage strategy",
            "Rate differentials can be noisy at 1-minute",
        ],
        "best_for": "Experimental use. Monitor signals before enabling auto-trade.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

def _chart_supertrend() -> go.Figure:
    df = _candles(120, seed=7, trend=0.0003)
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(closes)

    tr = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
          if i > 0 else highs[i]-lows[i] for i in range(n)]
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False).mean().values
    hl2 = (highs + lows) / 2
    bu, bl = hl2 + 3*atr, hl2 - 3*atr
    fu, fl = bu.copy(), bl.copy()
    direction = [1]*n
    for i in range(1, n):
        fu[i] = min(bu[i], fu[i-1]) if closes[i-1] <= fu[i-1] else bu[i]
        fl[i] = max(bl[i], fl[i-1]) if closes[i-1] >= fl[i-1] else bl[i]
        prev = direction[i-1]
        if prev == -1:
            direction[i] = 1 if closes[i] > fu[i] else -1
        else:
            direction[i] = -1 if closes[i] < fl[i] else 1
    direction = np.array(direction)

    st_line = np.where(direction == 1, fl, fu)
    bull_x = [i for i in range(n) if direction[i] == 1]
    bear_x = [i for i in range(n) if direction[i] == -1]

    fig = _fig_base()
    _add_candle_trace(fig, df)
    fig.add_trace(go.Scatter(x=bull_x, y=st_line[bull_x], mode="lines",
        line=dict(color=C_GREEN, width=2), name="Supertrend ↑"))
    fig.add_trace(go.Scatter(x=bear_x, y=st_line[bear_x], mode="lines",
        line=dict(color=C_RED, width=2), name="Supertrend ↓"))

    # Mark flips
    for i in range(1, n):
        if direction[i] != direction[i-1]:
            color = C_GREEN if direction[i] == 1 else C_RED
            label = "BUY" if direction[i] == 1 else "SELL"
            fig.add_annotation(x=i, y=closes[i], text=label,
                font=dict(color=color, size=10), showarrow=True,
                arrowcolor=color, arrowhead=2, ay=-25 if direction[i]==1 else 25)
    return fig


def _chart_rsi() -> go.Figure:
    df = _candles(120, seed=3, trend=0.0)
    closes = df["Close"].astype(float)
    delta = closes.diff()
    avg_g = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    avg_l = (-delta).clip(lower=0).ewm(com=13, min_periods=14).mean()
    rsi   = 100 - (100 / (1 + avg_g / avg_l.replace(0, float("nan"))))

    fig = _fig_base(rows=2, row_heights=[0.6, 0.4])
    _add_candle_trace(fig, df, row=1)
    fig.add_trace(go.Scatter(y=rsi, line=dict(color=C_BLUE, width=1.5), name="RSI(14)"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color=C_RED,   width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color=C_GREEN, width=1, dash="dash"), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=C_RED,   opacity=0.05, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=C_GREEN, opacity=0.05, row=2, col=1)
    fig.add_annotation(x=len(rsi)-5, y=73, text="Overbought 70",
        font=dict(color=C_RED, size=9), showarrow=False, row=2, col=1)
    fig.add_annotation(x=len(rsi)-5, y=27, text="Oversold 30",
        font=dict(color=C_GREEN, size=9), showarrow=False, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2)
    fig.update_layout(height=380)
    return fig


def _chart_macd() -> go.Figure:
    df = _candles(120, seed=5, trend=0.0002)
    closes = df["Close"].astype(float)
    fast = closes.ewm(span=12, adjust=False).mean()
    slow = closes.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    sig  = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig

    colors = [C_GREEN if v >= 0 else C_RED for v in hist]
    fig = _fig_base(rows=2, row_heights=[0.55, 0.45])
    _add_candle_trace(fig, df, row=1)
    fig.add_trace(go.Bar(y=hist, marker_color=colors, name="Histogram", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(y=macd, line=dict(color=C_BLUE,   width=1.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(y=sig,  line=dict(color=C_ORANGE, width=1.5, dash="dot"), name="Signal"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=C_MUTED, width=0.5), row=2, col=1)
    fig.update_layout(height=380)
    return fig


def _chart_bollinger() -> go.Figure:
    df = _candles(120, seed=9, trend=0.0)
    closes = df["Close"].astype(float)
    sma = closes.rolling(20).mean()
    std = closes.rolling(20).std()
    upper = sma + 2*std
    lower = sma - 2*std
    width = (upper - lower) / sma
    squeeze = width <= width.quantile(0.20)

    fig = _fig_base(rows=2, row_heights=[0.65, 0.35])
    _add_candle_trace(fig, df, row=1)
    x = list(range(len(closes)))
    fig.add_trace(go.Scatter(x=x+x[::-1],
        y=list(upper)+list(lower[::-1]),
        fill="toself", fillcolor="rgba(77,159,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="BB Band", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(y=upper, line=dict(color=C_BLUE, width=1, dash="dot"), name="Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(y=lower, line=dict(color=C_BLUE, width=1, dash="dot"), name="Lower", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=sma,   line=dict(color=C_MUTED, width=1), name="SMA(20)"), row=1, col=1)

    sq_color = [C_YELLOW if s else "rgba(0,0,0,0)" for s in squeeze]
    fig.add_trace(go.Bar(y=width*100, marker_color=sq_color, name="BB Width (squeeze=yellow)", opacity=0.8), row=2, col=1)
    fig.update_layout(height=400)
    return fig


def _chart_ichimoku() -> go.Figure:
    df = _candles(200, seed=11, trend=0.0002)
    highs  = df["High"].astype(float)
    lows   = df["Low"].astype(float)
    closes = df["Close"].astype(float)

    def dm(h, l, p): return (h.rolling(p).max() + l.rolling(p).min()) / 2
    tenkan = dm(highs, lows, 9)
    kijun  = dm(highs, lows, 26)
    span_a = (tenkan + kijun) / 2
    span_b = dm(highs, lows, 52)

    n = len(closes)
    x = list(range(n))
    fig = _fig_base()
    _add_candle_trace(fig, df)
    fig.add_trace(go.Scatter(x=x+x[::-1],
        y=list(span_a)+list(span_b[::-1]),
        fill="toself",
        fillcolor="rgba(34,211,238,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="Kumo Cloud"), row=1, col=1)
    fig.add_trace(go.Scatter(y=span_a, line=dict(color=C_TEAL,   width=1), name="Span A"), row=1, col=1)
    fig.add_trace(go.Scatter(y=span_b, line=dict(color=C_PURPLE, width=1), name="Span B"), row=1, col=1)
    fig.add_trace(go.Scatter(y=tenkan, line=dict(color=C_ORANGE, width=1, dash="dot"), name="Tenkan"), row=1, col=1)
    fig.add_trace(go.Scatter(y=kijun,  line=dict(color=C_BLUE,   width=1, dash="dot"), name="Kijun"),  row=1, col=1)
    return fig


def _chart_ma_crossover() -> go.Figure:
    df = _candles(120, seed=13, trend=0.0002)
    closes = df["Close"].astype(float)
    ema9  = closes.ewm(span=9,  adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    cross_up   = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
    cross_down = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))

    fig = _fig_base()
    _add_candle_trace(fig, df)
    fig.add_trace(go.Scatter(y=ema9,  line=dict(color=C_ORANGE, width=1.5), name="EMA 9"))
    fig.add_trace(go.Scatter(y=ema21, line=dict(color=C_BLUE,   width=1.5), name="EMA 21"))
    for i, v in enumerate(cross_up):
        if v:
            fig.add_annotation(x=i, y=float(closes.iloc[i]), text="BUY",
                font=dict(color=C_GREEN, size=10), arrowcolor=C_GREEN,
                arrowhead=2, showarrow=True, ay=-25)
    for i, v in enumerate(cross_down):
        if v:
            fig.add_annotation(x=i, y=float(closes.iloc[i]), text="SELL",
                font=dict(color=C_RED, size=10), arrowcolor=C_RED,
                arrowhead=2, showarrow=True, ay=25)
    return fig


def _chart_candlestick_patterns() -> go.Figure:
    """Show a couple of textbook candlestick patterns on a synthetic chart."""
    # Hand-craft a bullish engulfing + bearish engulfing pattern
    rng = np.random.default_rng(99)
    n = 40
    o = [1.0000]
    c = [0.9980]
    for _ in range(n - 1):
        prev_c = c[-1]
        delta = rng.normal(0, 0.003)
        new_c = prev_c * (1 + delta)
        new_o = prev_c
        o.append(new_o); c.append(new_c)
    h = [max(oi, ci) * (1 + abs(rng.normal(0, 0.001))) for oi, ci in zip(o, c)]
    l = [min(oi, ci) * (1 - abs(rng.normal(0, 0.001))) for oi, ci in zip(o, c)]
    # Inject bullish engulfing at i=15
    o[14], c[14] = 1.005, 0.998   # bearish candle
    o[15], c[15] = 0.997, 1.008   # large bullish engulfs it
    h[15], l[15] = 1.010, 0.995
    # Inject bearish engulfing at i=30
    o[29], c[29] = 0.998, 1.004   # bullish candle
    o[30], c[30] = 1.006, 0.994   # large bearish engulfs it
    h[30], l[30] = 1.008, 0.992
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})

    fig = _fig_base()
    _add_candle_trace(fig, df)
    fig.add_annotation(x=15, y=l[15]-0.003, text="Bullish Engulfing\n▲ BUY",
        font=dict(color=C_GREEN, size=10), showarrow=True, arrowcolor=C_GREEN,
        arrowhead=2, ay=30)
    fig.add_annotation(x=30, y=h[30]+0.003, text="Bearish Engulfing\n▼ SELL",
        font=dict(color=C_RED, size=10), showarrow=True, arrowcolor=C_RED,
        arrowhead=2, ay=-30)
    return fig


_CHART_FNS: dict[str, object] = {
    "supertrend":        _chart_supertrend,
    "rsi":               _chart_rsi,
    "macd":              _chart_macd,
    "bollinger_squeeze": _chart_bollinger,
    "ichimoku":          _chart_ichimoku,
    "ma_crossover":      _chart_ma_crossover,
    "candlestick":       _chart_candlestick_patterns,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.subheader("Strategy Reference")
    st.caption("Select a strategy to see how it works, what signals it generates, and when to use it.")

    meta_order = list(STRATEGY_META.keys())
    display_labels = [
        f"{STRATEGY_META[k]['icon']}  {STRATEGY_META[k]['display_name']}"
        for k in meta_order
    ]
    chosen_label = st.selectbox(
        "Strategy", display_labels,
        label_visibility="collapsed",
        help="Choose a strategy to explore",
    )
    key = meta_order[display_labels.index(chosen_label)]
    meta = STRATEGY_META[key]

    st.divider()

    # ── Header ───────────────────────────────────────────────────────────────
    cat_colors = {
        "AI": C_PURPLE, "Trend-Following": C_BLUE,
        "Breakout": C_ORANGE, "Oscillator": C_GREEN,
        "Price Action": C_YELLOW, "Arbitrage": C_TEAL,
    }
    cat_color = cat_colors.get(meta["category"], C_MUTED)

    st.markdown(
        f"""
        <div style="margin-bottom:4px">
          <span style="font-size:1.8rem;font-weight:800;letter-spacing:-0.5px">{meta['icon']} {meta['display_name']}</span>
          &nbsp;&nbsp;
          <span style="background:{cat_color}22;color:{cat_color};padding:3px 10px;
                border-radius:999px;font-size:0.75rem;font-weight:600;
                border:1px solid {cat_color}44">{meta['category']}</span>
        </div>
        <p style="color:#aaaacc;font-size:1.05rem;margin:0 0 12px 0">{meta['tagline']}</p>
        """,
        unsafe_allow_html=True,
    )

    # ── Chart + How it works (side by side on wide screens) ──────────────────
    has_chart = key in _CHART_FNS
    if has_chart:
        col_doc, col_chart = st.columns([1, 1], gap="large")
    else:
        col_doc = st.container()

    with col_doc:
        st.markdown("#### How it works")
        for i, step in enumerate(meta["how_it_works"], 1):
            st.markdown(f"{i}. {step}")

    if has_chart:
        with col_chart:
            st.markdown("#### Example chart")
            fig = _CHART_FNS[key]()
            st.plotly_chart(fig, use_container_width=True, config=_PLOTLY_CFG)
    elif key == "llm":
        st.info(
            "📸 The LLM strategy sends a **rendered chart image** to GPT-4o Vision. "
            "No fixed indicator lines — the model interprets the candlesticks visually, "
            "exactly as a human trader would when reading a chart."
        )

    st.divider()

    # ── Signal rules ─────────────────────────────────────────────────────────
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    signals = meta["signals"]
    with sig_col1:
        st.markdown(
            f"""<div style="background:#00c89615;border:1px solid #00c89640;border-radius:8px;
                padding:14px 16px">
                <div style="color:{C_GREEN};font-weight:700;font-size:0.85rem;
                     letter-spacing:1px;margin-bottom:6px">🟢 BUY</div>
                <div style="color:{C_TEXT};font-size:0.88rem">{signals['BUY']}</div>
            </div>""", unsafe_allow_html=True)
    with sig_col2:
        st.markdown(
            f"""<div style="background:#ff4d6d15;border:1px solid #ff4d6d40;border-radius:8px;
                padding:14px 16px">
                <div style="color:{C_RED};font-weight:700;font-size:0.85rem;
                     letter-spacing:1px;margin-bottom:6px">🔴 SELL</div>
                <div style="color:{C_TEXT};font-size:0.88rem">{signals['SELL']}</div>
            </div>""", unsafe_allow_html=True)
    with sig_col3:
        st.markdown(
            f"""<div style="background:#88888815;border:1px solid #88888840;border-radius:8px;
                padding:14px 16px">
                <div style="color:{C_MUTED};font-weight:700;font-size:0.85rem;
                     letter-spacing:1px;margin-bottom:6px">⚪ HOLD</div>
                <div style="color:{C_TEXT};font-size:0.88rem">{signals['HOLD']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Parameters, Pros/Cons, Best For ──────────────────────────────────────
    p_col, pc_col = st.columns([1, 1], gap="large")

    with p_col:
        st.markdown("#### Parameters")
        rows_html = "".join(
            f"""<tr>
              <td style="padding:7px 12px;color:{C_BLUE};font-weight:600;
                   font-family:monospace;white-space:nowrap">{p[0]}</td>
              <td style="padding:7px 12px;color:{C_YELLOW};font-weight:600">{p[1]}</td>
              <td style="padding:7px 12px;color:{C_MUTED};font-size:0.85rem">{p[2]}</td>
            </tr>"""
            for p in meta["parameters"]
        )
        st.markdown(
            f"""<table style="width:100%;border-collapse:collapse;
                 background:#0d1117;border-radius:8px;overflow:hidden">
              <thead><tr>
                <th style="padding:8px 12px;color:{C_MUTED};font-size:0.75rem;
                     text-align:left;border-bottom:1px solid {C_GRID}">PARAMETER</th>
                <th style="padding:8px 12px;color:{C_MUTED};font-size:0.75rem;
                     text-align:left;border-bottom:1px solid {C_GRID}">DEFAULT</th>
                <th style="padding:8px 12px;color:{C_MUTED};font-size:0.75rem;
                     text-align:left;border-bottom:1px solid {C_GRID}">DESCRIPTION</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )

    with pc_col:
        st.markdown("#### Pros & Cons")
        pros_html = "".join(
            f'<li style="color:{C_TEXT};font-size:0.88rem;margin-bottom:5px">'
            f'<span style="color:{C_GREEN}">✓</span> {p}</li>'
            for p in meta["pros"]
        )
        cons_html = "".join(
            f'<li style="color:{C_TEXT};font-size:0.88rem;margin-bottom:5px">'
            f'<span style="color:{C_RED}">✗</span> {p}</li>'
            for p in meta["cons"]
        )
        st.markdown(
            f'<ul style="margin:0 0 10px 0;padding-left:4px;list-style:none">{pros_html}</ul>'
            f'<ul style="margin:0;padding-left:4px;list-style:none">{cons_html}</ul>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Best For ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:{cat_color}0f;border-left:3px solid {cat_color};
             border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:8px">
          <span style="color:{cat_color};font-weight:700;font-size:0.8rem;
               letter-spacing:1px">BEST USED FOR</span><br>
          <span style="color:{C_TEXT};font-size:0.92rem">{meta['best_for']}</span>
        </div>""",
        unsafe_allow_html=True,
    )
