"""Modal dialog to create a custom bot in instruments.toml."""
from __future__ import annotations

import logging

import streamlit as st

import instrument_config
import strategies
import trading_engine

log = logging.getLogger(__name__)

_INTERVAL_OPTIONS: list[tuple[int, str]] = [
    (secs, instrument_config.interval_label_for_secs(secs))
    for secs in instrument_config.SUPPORTED_INTERVAL_SECS
]


def _instrument_choices(all_instruments: dict[str, int]) -> list[str]:
    """Known fleet assets first, then any other eToro labels."""
    known = list(instrument_config.KNOWN_ASSET_LABELS.values())
    opts: list[str] = [lbl for lbl in known if lbl in all_instruments or not all_instruments]
    seen = set(opts)
    for lbl in sorted(all_instruments):
        if lbl not in seen:
            opts.append(lbl)
            seen.add(lbl)
    return opts


def _strategy_choices() -> list[tuple[str, str]]:
    """All registered strategies; LLM is async (Visual Bot) but fully supported live."""
    names = strategies.display_names()
    out: list[tuple[str, str]] = [
        (strat.key, names.get(strat.key, strat.display_name or strat.key))
        for strat in strategies.all_strategies()
    ]
    # LLM first (default strategy), then alphabetical by display name
    out.sort(key=lambda x: (0 if x[0] == "llm" else 1, x[1].lower()))
    return out


def create_custom_bot(
    *,
    label: str,
    strategy: str,
    interval_secs: int,
    check_in_secs: int,
    demo_amount: float,
    candle_count: int,
    auto_trade: bool,
    all_instruments: dict[str, int],
    api_key: str,
    user_key: str,
    is_demo: bool,
) -> tuple[bool, str]:
    """Append bot to instruments.toml and start its engine thread."""
    label = (label or "").strip()
    if not label:
        return False, "Select an instrument."

    if label not in instrument_config.KNOWN_FLEET_LABELS and label not in all_instruments:
        return False, f"Instrument not found on eToro: {label}"

    specs = instrument_config.load_specs(enabled_only=False)
    existing_keys = {s.key for s in specs}
    interval = instrument_config.interval_label_for_secs(interval_secs)
    ci = int(check_in_secs) if check_in_secs else int(interval_secs)
    key = instrument_config.suggest_bot_key(
        label, strategy, interval_secs, existing_keys,
    )

    try:
        spec = instrument_config.append_bot(
            key=key,
            label=label,
            strategy=strategy,
            interval=interval,
            interval_secs=interval_secs,
            candle_count=int(candle_count),
            demo_amount=float(demo_amount),
            auto_trade=bool(auto_trade),
            created_via=instrument_config.BOT_SOURCE_CUSTOM,
            check_in_secs=ci,
        )
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        log.exception("Custom bot create failed")
        return False, f"Could not save bot: {exc}"

    instrument_config.invalidate_cache()
    resolved = instrument_config.resolve_ids([spec], all_instruments)
    if not resolved:
        return False, (
            f"Bot saved as `{key}` but eToro could not resolve **{label}**. "
            "Check the instrument name or API keys."
        )

    try:
        trading_engine.start_instrument(
            resolved[0],
            api_key=api_key,
            user_key=user_key,
            is_demo=is_demo,
        )
    except Exception as exc:
        log.warning("Engine start after custom bot create failed", exc_info=True)
        return True, (
            f"Created **`{key}`** ({strategy} · {interval} · {label}) but engine "
            f"start failed: {exc}. Toggle the bot ON on the Bots tab."
        )

    at_note = " Auto-trade is **ON**." if auto_trade else " Auto-trade is **OFF**."
    ci_lbl = instrument_config.interval_label_for_secs(
        instrument_config.effective_check_in_secs(
            key, interval_secs, toml_check_in=spec.check_in_secs,
        )
    )
    return True, (
        f"Created **`{key}`** — {strategy} on **{label}** · trade **{interval}** · "
        f"check-in **{ci_lbl}** (${demo_amount:,.0f}/trade).{at_note}"
    )


def _close_create_bot_dialog() -> None:
    st.session_state["_create_bot_dialog_open"] = False


def _open_create_bot_dialog() -> None:
    st.session_state["_create_bot_dialog_open"] = True


@st.dialog("Create custom bot", width="large", on_dismiss=_close_create_bot_dialog)
def create_bot_dialog(
    *,
    all_instruments: dict[str, int],
    api_key: str,
    user_key: str,
    is_demo: bool,
) -> None:
    """Build-a-bot form — opens as a modal over the Bots tab."""
    st.caption(
        "Pick an instrument, strategy, trade interval, and exit check-in. Saved to "
        "**instruments.toml**; the engine starts immediately."
    )

    inst_opts = _instrument_choices(all_instruments)
    if not inst_opts:
        st.warning("No instruments loaded — check eToro API keys and refresh the app.")
        if st.button("Close", key="create_bot_close_empty", on_click=_close_create_bot_dialog):
            pass
        return

    strat_opts = _strategy_choices()
    default_label = instrument_config.KNOWN_ASSET_LABELS.get("Bitcoin", inst_opts[0])
    if default_label not in inst_opts:
        default_label = inst_opts[0]

    # No st.form — widgets must rerun the dialog on change so the config-key
    # preview and check-in options stay in sync with trade interval / strategy.
    label = st.selectbox(
        "Instrument",
        options=inst_opts,
        index=inst_opts.index(default_label) if default_label in inst_opts else 0,
        key="create_bot_label",
        help="Hardcoded fleet assets are listed first. Scroll for more eToro instruments.",
    )

    c1, c2 = st.columns(2)
    with c1:
        strategy = st.selectbox(
            "Strategy",
            options=[k for k, _ in strat_opts],
            format_func=lambda k: strategies.display_names().get(k, k),
            key="create_bot_strategy",
            help="LLM (AI Vision) sends chart images to Visual Bot for analysis — "
                 "requires Visual Bot online. Rule-based strategies run in-process.",
        )
    with c2:
        interval_secs = st.selectbox(
            "Trade interval (entries)",
            options=[secs for secs, _ in _INTERVAL_OPTIONS],
            format_func=lambda s: instrument_config.interval_label_for_secs(s),
            index=3,  # 15 Minutes
            key="create_bot_interval",
        )

    _ci_opts = instrument_config.check_in_options(int(interval_secs))
    _ci_state_key = "create_bot_check_in"
    if st.session_state.get(_ci_state_key) not in _ci_opts:
        st.session_state[_ci_state_key] = _ci_opts[0]
    check_in_secs = st.selectbox(
        "Exit check-in interval",
        options=_ci_opts,
        format_func=lambda s: (
            f"{instrument_config.interval_label_for_secs(s)}"
            + (" (same as trade)" if s == int(interval_secs) else " ⚡ faster exits")
        ),
        key=_ci_state_key,
        help="How often reversal exits are re-checked. Finer intervals react "
             "faster (½ or ¼ of the trade interval when supported by eToro).",
    )

    c3, c4 = st.columns(2)
    with c3:
        demo_amount = st.number_input(
            "Demo trade size ($)",
            min_value=10.0,
            max_value=100_000.0,
            value=1000.0,
            step=50.0,
            key="create_bot_amount",
        )
    with c4:
        candle_count = st.number_input(
            "Candles in context",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            key="create_bot_candles",
        )

    auto_trade = st.checkbox(
        "Turn auto-trade ON immediately",
        value=False,
        key="create_bot_auto_trade",
        help="Off by default — same as fleet-created bots. Turn on later from the Bots tab.",
    )

    existing_keys = {
        s.key for s in instrument_config.load_specs(enabled_only=False)
    }
    preview_key = instrument_config.suggest_bot_key(
        label,
        strategy,
        int(interval_secs),
        existing_keys,
    )
    st.caption(f"Config key: `{preview_key}`")

    btn_create, btn_cancel = st.columns(2)
    with btn_create:
        create_clicked = st.button(
            "Create bot",
            type="primary",
            use_container_width=True,
            key="create_bot_submit",
        )
    with btn_cancel:
        if st.button(
            "Cancel",
            key="create_bot_cancel",
            use_container_width=True,
            on_click=_close_create_bot_dialog,
        ):
            pass

    if create_clicked:
        ok, msg = create_custom_bot(
            label=label,
            strategy=strategy,
            interval_secs=int(interval_secs),
            check_in_secs=int(check_in_secs),
            demo_amount=float(demo_amount),
            candle_count=int(candle_count),
            auto_trade=bool(auto_trade),
            all_instruments=all_instruments,
            api_key=api_key,
            user_key=user_key,
            is_demo=is_demo,
        )
        st.session_state["_create_bot_msg"] = msg
        st.session_state["_create_bot_ok"] = ok
        if ok:
            st.session_state["_create_bot_dialog_open"] = False
            st.rerun()
        else:
            st.error(msg)


def render_create_bot_section(
    *,
    all_instruments: dict[str, int],
    api_key: str,
    user_key: str,
    is_demo: bool,
) -> None:
    """Button + status on the Bots tab; opens the modal dialog."""
    msg = st.session_state.get("_create_bot_msg")
    ok_flag = st.session_state.get("_create_bot_ok", False)
    if msg:
        if ok_flag:
            st.success(msg)
        else:
            st.error(msg)
        st.session_state.pop("_create_bot_msg", None)
        st.session_state.pop("_create_bot_ok", None)

    with st.expander("Create your own bot", expanded=bool(msg)):
        st.caption(
            "Build a single bot — instrument, strategy, **trade interval**, and "
            "**exit check-in**. Opens a setup dialog; saved to **instruments.toml**."
        )
        st.button(
            "Open bot builder…",
            type="primary",
            key="bots_open_create_dialog",
            use_container_width=True,
            on_click=_open_create_bot_dialog,
        )

    if st.session_state.get("_create_bot_dialog_open"):
        create_bot_dialog(
            all_instruments=all_instruments,
            api_key=api_key,
            user_key=user_key,
            is_demo=is_demo,
        )
