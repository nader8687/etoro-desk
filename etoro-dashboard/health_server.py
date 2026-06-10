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
    }


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] in ("/health", "/"):
            body = json.dumps(_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

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
