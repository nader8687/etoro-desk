"""Centralised trading-mode policy — the single guard against accidental live trading.

This build executes orders on the eToro **demo** (virtual-money) endpoints only:
`etoro_client` exposes no real-money order method, and the engine refuses to
trade a non-demo account.  That guarantee was previously implicit (spread across
an engine `if not config.is_demo` and the absence of a live endpoint).  It now
lives here, named and asserted at the order boundary, so it cannot be eroded by
a future change without someone deliberately flipping the opt-in below.

To ever enable live trading you must do BOTH, on purpose:
  1. set the env var  ALLOW_LIVE_TRADING=1  (or true/yes), and
  2. implement real-money order methods in etoro_client and route them here.
Doing (1) alone changes nothing — there is still no live order path.  The env
gate exists so live cannot be reached by accident even after (2) is added.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def live_trading_allowed() -> bool:
    """True only when the operator has explicitly opted in via env."""
    return os.environ.get("ALLOW_LIVE_TRADING", "").strip().lower() in _TRUE


def mode_label() -> str:
    return "LIVE (opt-in)" if live_trading_allowed() else "DEMO-only"


def trade_allowed(is_demo: bool) -> bool:
    """Whether an order in this mode may be placed at all."""
    return bool(is_demo) or live_trading_allowed()


def assert_trade_allowed(is_demo: bool) -> None:
    """Raise PermissionError if a non-demo order is attempted without the
    explicit live opt-in.  Called at the order boundary (trade_manager) so the
    demo-only guarantee holds even if an upstream gate is removed."""
    if not trade_allowed(is_demo):
        raise PermissionError(
            "Live trading is disabled (demo-only build). Refusing a real-money "
            "order. Set ALLOW_LIVE_TRADING=1 and implement a live order path to "
            "enable — this is intentional protection against accidental live trades."
        )
