"""Shared UI styling, theme tokens, and layout helpers for EtoroDesk."""
from datetime import datetime, timezone

import streamlit as st

import timez

# ── Theme: balanced dark terminal (readable, not pitch-black) ───────────────
C_APP      = "#0f1216"
C_BG       = "#181c22"
C_SURFACE  = "#1f252e"
C_SURFACE2 = "#262d38"
C_BORDER   = "#343d4d"
C_BORDER2  = "#434d5f"

C_TEXT     = "#f0f2f5"
C_TEXT2    = "#c5cdd8"
C_MUTED    = "#939dad"
C_LABEL    = "#7a8494"

C_UP       = "#3dba9c"
C_DOWN     = "#e05d5d"
C_HOLD     = "#c9a227"
C_ACCENT   = "#6b9eff"
C_LIVE     = "#4db88a"
C_SUCCESS  = "#3dba9c"
C_WARN     = "#d4a843"

C_GRID     = "#232a34"
C_INPUT    = "#262d38"
C_UP_RGBA  = "rgba(61,186,156,0.22)"
C_DOWN_RGBA = "rgba(224,93,93,0.20)"


def pnl_color(value: float | None) -> str:
    if value is None:
        return C_MUTED
    return C_UP if value >= 0 else C_DOWN


def signal_color(signal: str) -> str:
    return {"BUY": C_UP, "SELL": C_DOWN, "HOLD": C_HOLD}.get(signal.upper(), C_MUTED)


def direction_color(direction: str) -> str:
    return C_UP if direction.upper() == "LONG" else C_DOWN


def _widget_css() -> str:
    """Streamlit native widgets — kill white inputs & chrome."""
    return f"""
    /* ── Hide / darken Streamlit chrome ─────────────────────────────── */
    [data-testid="stHeader"] {{
        background: {C_APP} !important;
        border-bottom: 1px solid {C_BORDER};
    }}
    [data-testid="stHeader"] button {{
        color: {C_TEXT2} !important;
    }}
    [data-testid="stToolbar"] {{
        background: {C_APP} !important;
    }}
    [data-testid="stDecoration"] {{
        display: none;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .stApp {{
        background: {C_APP};
    }}

    /* ── Labels & captions (main + sidebar) ─────────────────────────── */
    label, .stSelectbox label, .stNumberInput label, .stTextInput label,
    .stDateInput label, .stRadio label, .stToggle label,
    .stCheckbox label, [data-testid="stWidgetLabel"] p {{
        color: {C_TEXT2} !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {C_TEXT2} !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {{
        color: {C_TEXT} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color: {C_MUTED} !important;
        font-size: 0.76rem !important;
    }}

    /* ── Text inputs, selects, number fields ──────────────────────── */
    input, textarea, select,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span,
    .stNumberInput input,
    .stTextInput input,
    .stDateInput input {{
        background-color: {C_INPUT} !important;
        color: {C_TEXT} !important;
        border-color: {C_BORDER} !important;
        border-radius: 6px !important;
    }}
    div[data-baseweb="select"] > div {{
        border: 1px solid {C_BORDER} !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {C_MUTED} !important;
    }}
    [data-baseweb="popover"] {{
        background: {C_SURFACE} !important;
        border: 1px solid {C_BORDER} !important;
    }}
    [data-baseweb="menu"] li {{
        background: {C_SURFACE} !important;
        color: {C_TEXT} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background: {C_SURFACE2} !important;
    }}

    /* ── Radio & toggle ─────────────────────────────────────────────── */
    .stRadio div[role="radiogroup"] label {{
        background: {C_INPUT} !important;
        border: 1px solid {C_BORDER} !important;
        color: {C_TEXT2} !important;
        padding: 6px 14px !important;
        border-radius: 6px !important;
    }}
    .stRadio div[role="radiogroup"] label[data-checked="true"],
    .stRadio div[role="radiogroup"] label:has(input:checked) {{
        background: rgba(107,158,255,0.15) !important;
        border-color: {C_ACCENT} !important;
        color: {C_TEXT} !important;
    }}
    .stToggle [data-testid="stMarkdownContainer"] p {{
        color: {C_TEXT2} !important;
        font-weight: 600 !important;
    }}

    /* ── Buttons (clear affordance: depth, border, contrast) ─────────── */
    .stButton, [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {{
        width: 100%;
    }}
    .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"],
    button[kind="secondary"],
    button[kind="primary"] {{
        border-radius: 7px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.01em !important;
        min-height: 2.1rem !important;
        padding: 0.35rem 0.85rem !important;
        cursor: pointer !important;
        transition: background 0.15s, border-color 0.15s, box-shadow 0.15s !important;
    }}
    /* Secondary / default action */
    .stButton > button,
    [data-testid="baseButton-secondary"],
    button[kind="secondary"] {{
        background: linear-gradient(180deg, #343e4c 0%, #2c3542 100%) !important;
        border: 1px solid {C_BORDER2} !important;
        color: {C_TEXT} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35),
                    inset 0 1px 0 rgba(255,255,255,0.07) !important;
    }}
    .stButton > button:hover:not(:disabled),
    [data-testid="baseButton-secondary"]:hover:not(:disabled),
    button[kind="secondary"]:hover:not(:disabled) {{
        background: linear-gradient(180deg, #3d4858 0%, #343e4c 100%) !important;
        border-color: #5a677a !important;
        color: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.09) !important;
    }}
    /* Primary action */
    .stButton > button[kind="primary"],
    [data-testid="baseButton-primary"],
    button[kind="primary"] {{
        background: linear-gradient(180deg, #6b9eff 0%, #5289e8 100%) !important;
        border: 1px solid #84afff !important;
        color: #fff !important;
        box-shadow: 0 2px 6px rgba(82,137,232,0.45),
                    inset 0 1px 0 rgba(255,255,255,0.15) !important;
    }}
    .stButton > button[kind="primary"]:hover:not(:disabled),
    [data-testid="baseButton-primary"]:hover:not(:disabled),
    button[kind="primary"]:hover:not(:disabled) {{
        background: linear-gradient(180deg, #7aabff 0%, #6b9eff 100%) !important;
        border-color: #9ec2ff !important;
        box-shadow: 0 3px 10px rgba(82,137,232,0.55) !important;
    }}
    /* Disabled — still looks like a button */
    .stButton > button:disabled,
    [data-testid="baseButton-secondary"]:disabled,
    [data-testid="baseButton-primary"]:disabled {{
        opacity: 1 !important;
        background: {C_INPUT} !important;
        color: {C_MUTED} !important;
        border: 1px solid {C_BORDER} !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }}

    /* ── Alerts ───────────────────────────────────────────────────── */
    [data-testid="stAlert"] {{
        border-radius: 6px;
        border: 1px solid {C_BORDER};
        background: {C_SURFACE};
        color: {C_TEXT2};
    }}

    /* ── Compact inline toolbars (History date row, etc.) ─────────── */
    [data-testid="stSegmentedControl"] {{
        min-height: 0 !important;
    }}
    [data-testid="stSegmentedControl"] > div {{
        min-height: 0 !important;
        padding: 0 !important;
    }}
    [data-testid="stSegmentedControl"] button {{
        min-height: 2.1rem !important;
        padding: 0.3rem 0.75rem !important;
        font-size: 0.8rem !important;
    }}
    [data-testid="stDateInput"] > div {{
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }}
    [data-testid="stDateInput"] input {{
        min-height: 2.1rem !important;
        padding: 0.3rem 0.5rem !important;
        font-size: 0.84rem !important;
    }}
    """


def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {{
        --app: {C_APP};
        --bg: {C_BG};
        --surface: {C_SURFACE};
        --surface2: {C_SURFACE2};
        --border: {C_BORDER};
        --text: {C_TEXT};
        --text2: {C_TEXT2};
        --muted: {C_MUTED};
        --input: {C_INPUT};
        --up: {C_UP};
        --down: {C_DOWN};
        --accent: {C_ACCENT};
    }}

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {C_TEXT2};
    }}
    [data-testid="stAppViewContainer"] {{
        background: {C_APP};
    }}
    .main .block-container {{
        padding-top: 1rem;
        max-width: 100%;
        color: {C_TEXT2};
    }}

    {_widget_css()}

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {C_SURFACE} !important;
        border-right: 1px solid {C_BORDER};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: {C_SURFACE} !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.25rem;
    }}
    .side-section-title {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {C_MUTED};
        margin: 0 0 10px 0;
    }}
    .side-stat-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        font-size: 0.84rem;
        color: {C_TEXT2};
        border-bottom: 1px solid {C_BORDER};
    }}
    .side-stat-row:last-child {{ border-bottom: none; }}
    .side-stat-row .val {{ color: {C_TEXT}; font-weight: 600; }}

    /* ── Metrics ──────────────────────────────────────────────────── */
    div[data-testid="metric-container"] {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: {C_TEXT} !important;
    }}
    div[data-testid="stMetricLabel"] {{ color: {C_MUTED} !important; }}

    /* ── Header bar ───────────────────────────────────────────────── */
    .desk-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 12px;
    }}
    .desk-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {C_TEXT};
        letter-spacing: -0.02em;
        margin: 0;
    }}
    .desk-sub {{
        font-size: 0.8rem;
        color: {C_MUTED};
        margin-top: 3px;
    }}
    .pill-row {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .pill {{
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }}
    .pill-demo   {{ background: rgba(107,158,255,0.12); color: {C_ACCENT}; border-color: rgba(107,158,255,0.28); }}
    .pill-real   {{ background: rgba(201,162,39,0.1); color: {C_HOLD}; border-color: rgba(201,162,39,0.25); }}
    .pill-live   {{ background: rgba(77,184,138,0.12); color: {C_LIVE}; border-color: rgba(77,184,138,0.28); }}
    .pill-off    {{ background: {C_INPUT}; color: {C_MUTED}; border-color: {C_BORDER}; }}
    .pill-trade  {{ background: rgba(61,186,156,0.12); color: {C_UP}; border-color: rgba(61,186,156,0.28); }}

    /* ── Navigation (segmented control) ───────────────────────────── */
    [data-testid="stSegmentedControl"] {{
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        padding: 6px 8px;
        margin-bottom: 12px;
        overflow: visible !important;
    }}
    /* The control is a single flex row that CLIPS overflowing items by default —
       with 12 nav entries the row can overflow and tabs silently disappear.
       Let it wrap onto extra lines so every page is always reachable. */
    [data-testid="stSegmentedControl"] > div,
    [data-testid="stSegmentedControl"] [role="radiogroup"],
    [data-testid="stSegmentedControl"] [data-baseweb="button-group"] {{
        flex-wrap: wrap !important;
        row-gap: 4px !important;
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
    }}
    [data-testid="stSegmentedControl"] button {{
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: {C_MUTED} !important;
        border-radius: 6px !important;
    }}
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {{
        background: {C_SURFACE2} !important;
        color: {C_TEXT} !important;
        border: 1px solid {C_BORDER2} !important;
    }}

    /* ── Panels & bordered containers ─────────────────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: {C_SURFACE} !important;
        border-color: {C_BORDER} !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        margin-bottom: 10px !important;
    }}
    .panel-label {{
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {C_MUTED};
        margin: 0 0 8px 0;
    }}
    .toolbar-hint {{
        font-size: 0.78rem;
        color: {C_MUTED};
        margin: 0;
        padding-top: 4px;
    }}
    .toolbar-hint b {{ color: {C_TEXT2}; }}

    hr {{ border-color: {C_BORDER} !important; margin: 0.75rem 0 !important; }}

    /* ── Feed / status / quote / position cards ───────────────────── */
    .feed-status {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px 14px;
        font-size: 0.78rem;
        color: {C_MUTED};
        padding: 8px 12px;
        background: {C_SURFACE};
        border: 1px solid {C_BORDER};
        border-radius: 6px;
        margin-top: 6px;
    }}
    .feed-status b {{ color: {C_TEXT2}; font-weight: 600; }}
    .feed-dot-live {{ color: {C_LIVE}; }}

    .status-banner {{
        font-size: 0.8rem;
        line-height: 1.5;
        padding: 10px 12px;
        border-radius: 6px;
        border: 1px solid transparent;
        margin: 0;
    }}
    .status-manage {{
        background: rgba(201,162,39,0.08);
        border-color: rgba(201,162,39,0.2);
        color: #d4bc6a;
    }}
    .status-hunt {{
        background: rgba(61,186,156,0.08);
        border-color: rgba(61,186,156,0.22);
        color: {C_UP};
    }}
    .status-off {{
        background: {C_INPUT};
        border-color: {C_BORDER};
        color: {C_MUTED};
    }}

    .quote-strip {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 6px;
    }}
    .quote-cell {{
        background: {C_INPUT};
        border: 1px solid {C_BORDER};
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }}
    .quote-cell .lbl {{
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: {C_MUTED};
    }}
    .quote-cell .val {{
        font-size: 0.92rem;
        font-weight: 700;
        color: {C_TEXT};
        font-variant-numeric: tabular-nums;
    }}

    .pos-card {{
        background: {C_INPUT};
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        padding: 12px 14px;
    }}
    .pos-card.short {{ border-left: 3px solid {C_DOWN}; }}
    .pos-card.long  {{ border-left: 3px solid {C_UP}; }}
    .pos-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .pos-dir {{ font-size: 0.92rem; font-weight: 700; }}
    .pos-dir.long  {{ color: {C_UP}; }}
    .pos-dir.short {{ color: {C_DOWN}; }}
    .pos-badge {{
        font-size: 0.62rem; font-weight: 600; padding: 2px 7px;
        border-radius: 4px; text-transform: uppercase;
    }}
    .pos-badge.gain {{ background: rgba(61,186,156,0.14); color: {C_UP}; }}
    .pos-badge.loss {{ background: rgba(224,93,93,0.14); color: {C_DOWN}; }}
    .pos-badge.flat {{ background: {C_SURFACE2}; color: {C_MUTED}; }}
    .pos-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }}
    .pos-stat .lbl {{
        font-size: 0.6rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; color: {C_LABEL};
    }}
    .pos-stat .val {{
        font-size: 0.82rem; font-weight: 600; color: {C_TEXT};
        font-variant-numeric: tabular-nums;
    }}
    .pos-pnl {{
        margin-top: 10px; padding-top: 10px;
        border-top: 1px solid {C_BORDER};
        display: flex; justify-content: space-between; align-items: baseline;
    }}
    .pos-pnl .amt {{ font-size: 1.05rem; font-weight: 700; }}
    .pos-pnl .pct {{ font-size: 0.76rem; color: {C_MUTED}; }}

    .signal-card {{
        background: {C_INPUT};
        border-radius: 8px;
        padding: 12px 16px;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }}
    .signal-card.buy  {{ border: 1px solid rgba(61,186,156,0.35); }}
    .signal-card.sell {{ border: 1px solid rgba(224,93,93,0.35); }}
    .signal-card.hold {{ border: 1px solid rgba(201,162,39,0.3); }}
    .signal-action {{ font-size: 1.4rem; font-weight: 800; letter-spacing: 0.06em; }}
    .signal-meta {{ color: {C_TEXT2}; font-size: 0.8rem; line-height: 1.5; }}

    .exit-card {{
        background: {C_INPUT};
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 6px;
    }}
    .exit-card.hold  {{ border: 1px solid rgba(61,186,156,0.25); }}
    .exit-card.close {{ border: 1px solid rgba(201,162,39,0.28); }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}
    div.dvn-scroller {{ background: {C_SURFACE} !important; }}

    /* ── Portfolio table (eToro-style) ─────────────────────────────── */
    .pf-wrap {{
        border: 1px solid {C_BORDER};
        border-radius: 8px;
        overflow: hidden;
        background: {C_SURFACE};
        margin-top: 8px;
    }}
    .pf-section-title {{
        font-size: 0.92rem;
        font-weight: 700;
        color: {C_TEXT};
        margin: 0 0 8px 0;
    }}
    .pf-hist-stats {{
        display: flex;
        flex-wrap: wrap;
        gap: 20px 28px;
        margin: 0 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid {C_BORDER};
    }}
    .pf-stat-label {{
        margin: 0;
        font-size: 0.66rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {C_MUTED};
    }}
    .pf-stat-value {{
        margin: 2px 0 0 0;
        font-size: 1.02rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }}
    .pf-table {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }}
    .pf-table th.pf-th,
    .pf-table td {{
        padding: 7px 6px;
        vertical-align: middle;
    }}
    .pf-table th.pf-th-left,
    .pf-table td.pf-left {{
        text-align: left;
    }}
    .pf-table th.pf-th-right,
    .pf-table td.pf-right {{
        text-align: right;
    }}
    .pf-table th.pf-th-units-gap,
    .pf-table td.pf-col-units-gap {{
        padding-right: 2rem;
    }}
    .pf-table th.pf-th-opened,
    .pf-table td.pf-col-opened {{
        padding-left: 1rem;
        padding-right: 0.2rem;
    }}
    .pf-table th.pf-th-closed,
    .pf-table td.pf-col-closed {{
        padding-left: 0.2rem;
        padding-right: 0.35rem;
    }}
    .pf-table th.pf-th-price-gap,
    .pf-table td.pf-col-price-gap {{
        padding-right: 1.35rem;
    }}
    .pf-table th.pf-th-units-pos,
    .pf-table td.pf-col-units-pos {{
        padding-left: 1rem;
    }}
    .pf-table th.pf-th-bot,
    .pf-table td.pf-col-bot,
    .pf-table col.pf-col-bot {{
        min-width: 12.5rem;
    }}
    .pf-table th.pf-th-bot,
    .pf-table td.pf-col-bot {{
        white-space: nowrap;
    }}
    .pf-th {{
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {C_MUTED};
        margin: 0;
        border-bottom: 1px solid {C_BORDER};
    }}
    .pf-th.pf-sorted {{ color: {C_ACCENT}; }}
    .pf-row-hdr {{
        background: {C_SURFACE2};
        padding: 0 10px;
    }}
    .pf-row {{
        padding: 12px 10px;
        border-bottom: 1px solid {C_BORDER};
        align-items: center;
    }}
    .pf-row:last-child {{ border-bottom: none; }}
    .pf-symbol {{
        font-size: 0.88rem;
        font-weight: 700;
        color: {C_TEXT};
        line-height: 1.2;
    }}
    .pf-name {{
        font-size: 0.68rem;
        color: {C_MUTED};
        margin-top: 1px;
        line-height: 1.25;
    }}
    .pf-ts {{
        font-size: 0.72rem;
        color: {C_TEXT2};
        margin: 0;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .pf-bot-tag {{
        display: inline-block;
        font-size: 0.58rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {C_ACCENT};
        background: rgba(107,158,255,0.12);
        border: 1px solid rgba(107,158,255,0.25);
        border-radius: 3px;
        padding: 1px 5px;
        margin-left: 6px;
        vertical-align: middle;
    }}
    .pf-price {{
        font-size: 0.92rem;
        font-weight: 600;
        color: {C_TEXT};
        font-variant-numeric: tabular-nums;
    }}
    .pf-chg {{
        display: block;
        font-size: 0.72rem;
        font-weight: 500;
        margin-top: 2px;
        font-variant-numeric: tabular-nums;
    }}
    .pf-price, .pf-symbol, .pf-units, .pf-val, .pf-pnl, .pf-name, .pf-dir, .pf-ts {{
        margin: 0;
        padding: 0;
    }}
    .pf-chg.up {{ color: {C_UP}; }}
    .pf-chg.down {{ color: {C_DOWN}; }}
    .pf-units {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {C_TEXT};
        font-variant-numeric: tabular-nums;
    }}
    .pf-dir {{
        font-size: 0.68rem;
        font-weight: 600;
        margin-top: 2px;
    }}
    .pf-units-line .pf-dir {{
        display: inline;
        margin: 0 0 0 5px;
        font-size: 0.62rem;
        vertical-align: middle;
    }}
    .pf-dir.long {{ color: {C_UP}; }}
    .pf-dir.short {{ color: {C_DOWN}; }}
    .pf-val {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {C_TEXT};
        font-variant-numeric: tabular-nums;
    }}
    .pf-pnl {{
        font-size: 0.88rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }}
    [class*="st-key-pf_close_"] button {{
        background: {C_DOWN} !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }}
    [class*="st-key-pf_close_"] button:hover {{
        filter: brightness(1.08);
    }}
    </style>
    """, unsafe_allow_html=True)


def render_header(
    is_demo: bool,
    trading_active: bool = False,
    live_connected: bool = False,
) -> None:
    acct = "Demo" if is_demo else "Real"
    acct_cls = "pill-demo" if is_demo else "pill-real"
    live_cls = "pill-live" if live_connected else "pill-off"
    trade_cls = "pill-trade" if trading_active else "pill-off"
    trade_lbl = "Auto-Trade ON" if trading_active else "Auto-Trade OFF"
    live_lbl = "Feed Live" if live_connected else "Feed Off"

    st.markdown(f"""
    <div class="desk-header">
      <div>
        <p class="desk-title">EtoroDesk</p>
        <p class="desk-sub">AI-assisted trading desk · eToro Public API</p>
      </div>
      <div class="pill-row">
        <span class="pill {acct_cls}">{acct} Account</span>
        <span class="pill {live_cls}">{live_lbl}</span>
        <span class="pill {trade_cls}">{trade_lbl}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def panel_title(label: str) -> None:
    st.markdown(f'<p class="panel-label">{label}</p>', unsafe_allow_html=True)


def section_title(label: str) -> None:
    st.markdown(f'<p class="side-section-title">{label}</p>', unsafe_allow_html=True)


def status_banner(kind: str, message: str) -> None:
    cls = {"manage": "status-manage", "hunt": "status-hunt", "off": "status-off"}.get(
        kind, "status-off"
    )
    st.markdown(f'<p class="status-banner {cls}">{message}</p>', unsafe_allow_html=True)


def quote_strip(ask: float, bid: float, spread: float) -> None:
    st.markdown(
        f"""<div class="quote-strip">
          <div class="quote-cell"><div class="lbl">Ask</div>
            <div class="val">{ask:.5f}</div></div>
          <div class="quote-cell"><div class="lbl">Bid</div>
            <div class="val">{bid:.5f}</div></div>
          <div class="quote-cell"><div class="lbl">Spread</div>
            <div class="val">{spread:.5f}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def feed_status(
    badge: str,
    tick_age: float | None,
    tick_count: int,
    candle_count: int,
    *,
    engine: bool = False,
) -> None:
    live = "LIVE" in badge.upper()
    dot = f'<span class="feed-dot-live">●</span>' if live else ""
    age_txt = f"<b>{tick_age:.0f}s</b> ago" if tick_age is not None else "—"
    eng = " · <b>engine on</b>" if engine else ""
    st.markdown(
        f"""<div class="feed-status">
          {dot} <b>{badge.strip()}</b>
          · tick {age_txt}
          · <b>{tick_count}</b> ticks
          · <b>{candle_count}</b> candles{eng}
        </div>""",
        unsafe_allow_html=True,
    )


def render_sidebar_branding() -> None:
    st.markdown(f"""
    <div style="margin-bottom:2px">
      <span style="font-size:1.15rem;font-weight:700;color:{C_TEXT}">EtoroDesk</span>
    </div>
    <p style="font-size:0.8rem;color:{C_MUTED};margin:0 0 16px 0">Trading terminal</p>
    """, unsafe_allow_html=True)


def sidebar_status(
    visual_bot_ok: bool,
    positions_count: int = 0,
    *,
    engine_on: bool = False,
    trading_on: bool = False,
    feed_live: bool = False,
    ws_badge: str = "",
    auto_trade_count: int = 0,
) -> None:
    bot_col = C_LIVE if visual_bot_ok else C_DOWN
    bot_txt = "Online" if visual_bot_ok else "Offline"
    eng_col = C_LIVE if engine_on else C_MUTED
    eng_txt = "Running" if engine_on else "Stopped"
    feed_col = C_LIVE if feed_live else C_MUTED
    feed_txt = "Live" if feed_live else "Off"
    ts = f"{timez.now_str('%H:%M:%S')} {timez.abbrev()}"
    if auto_trade_count > 0:
        trade_col = C_LIVE
        trade_txt = f"{auto_trade_count} instrument{'s' if auto_trade_count != 1 else ''}"
    else:
        trade_col = C_MUTED
        trade_txt = "OFF"
    ws_line = (
        f"""<div class="side-stat-row">
          <span>WebSocket</span>
          <span class="val" style="font-size:0.78rem">{ws_badge}</span>
        </div>"""
        if ws_badge else ""
    )
    st.markdown(
        f"""<div class="side-stat-row">
          <span>Engine</span>
          <span class="val" style="color:{eng_col}">{eng_txt}</span>
        </div>
        <div class="side-stat-row">
          <span>Auto-trade</span>
          <span class="val" style="color:{trade_col}">{trade_txt}</span>
        </div>
        <div class="side-stat-row">
          <span>Feed</span>
          <span class="val" style="color:{feed_col}">{feed_txt}</span>
        </div>
        {ws_line}
        <div class="side-stat-row">
          <span>Visual Bot</span>
          <span class="val" style="color:{bot_col}">{bot_txt}</span>
        </div>
        <div class="side-stat-row">
          <span>Open positions</span>
          <span class="val">{positions_count}</span>
        </div>
        <div class="side-stat-row">
          <span>Updated</span>
          <span class="val" style="font-size:0.78rem">{ts}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def toolbar_hint(instrument: str, interval: str, feed_badge: str) -> None:
    st.markdown(
        f'<p class="toolbar-hint"><b>{instrument}</b> · {interval} · Feed: <b>{feed_badge}</b></p>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""<div style="
            text-align:center;padding:32px 16px;
            background:{C_SURFACE};border:1px dashed {C_BORDER};border-radius:10px;
        ">
          <div style="font-size:2rem;margin-bottom:8px">{icon}</div>
          <div style="font-weight:600;color:{C_TEXT};margin-bottom:4px">{title}</div>
          <div style="font-size:0.84rem;color:{C_MUTED}">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )
