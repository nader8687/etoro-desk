"""Lightweight JSON health endpoint (engine + WebSocket status)."""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)

_PORT = int(os.environ.get("DASHBOARD_HEALTH_PORT", "8502"))
_started = False


def _status() -> dict:
    import tick_manager
    import trade_manager
    import trading_engine

    cfg = trading_engine.get_config()
    iid = cfg.instrument_id if cfg else None
    ws_state = None
    if iid is not None:
        state = tick_manager.get_state(iid)
        ws_state = state.name if state is not None else None
    return {
        "status": "ok",
        "engine_running": trading_engine.is_running(),
        "trading_active": trading_engine.is_trading_active(),
        "instrument_id": iid,
        "ws_state": ws_state,
        "open_trades": len(trade_manager.get_all_open()),
    }


def _link_status() -> dict:
    """Per-bot linkage diagnostic (runs in the live Streamlit process)."""
    import json
    from pathlib import Path

    import bot_registry
    import positions_cache
    import trade_manager
    import trading_engine

    owners_path = Path(os.environ.get("POSITION_OWNERS_PATH", "/app/data/position_owners.json"))
    owners: dict[str, str] = {}
    if owners_path.exists():
        try:
            owners = json.loads(owners_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    registry = bot_registry.get_all()
    uuid_to_key = {v: k for k, v in registry.items()}
    cached = positions_cache.get_positions()
    open_pids = {str(p.get("position_id")) for p in cached if p.get("position_id") is not None}

    in_mem = {t.bot_id: t.etoro_position_id for t in trade_manager.get_all_open()}
    rows: list[dict] = []
    for pid, owner_uuid in owners.items():
        if open_pids and pid not in open_pids:
            continue
        key = uuid_to_key.get(owner_uuid, "")
        resolved = trading_engine.get_bot_uuid(key) if key else ""
        rows.append({
            "bot_key": key,
            "owner_uuid": owner_uuid[:8],
            "resolved_uuid": (resolved or "")[:8],
            "uuid_match": resolved == owner_uuid,
            "in_memory": trade_manager.has_open(resolved) if resolved else False,
            "position_id": pid,
        })

    unlinked = [r for r in rows if r["bot_key"] and not r["in_memory"]]
    return {
        "open_trades": len(in_mem),
        "cached_positions": len(cached),
        "owned_open_rows": len(rows),
        "unlinked_bots": len(unlinked),
        "unlinked_sample": unlinked[:10],
        "uuid_mismatches": [r for r in rows if r["bot_key"] and not r["uuid_match"]][:10],
    }


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            body = json.dumps(_status()).encode()
        elif path == "/health/links":
            body = json.dumps(_link_status()).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


def start_background() -> None:
    global _started
    if _started:
        return
    _started = True

    def _run() -> None:
        try:
            srv = HTTPServer(("0.0.0.0", _PORT), _Handler)
            log.info("Health server listening on :%s/health", _PORT)
            srv.serve_forever()
        except Exception:
            log.exception("Health server failed")

    threading.Thread(target=_run, daemon=True, name="health-server").start()
