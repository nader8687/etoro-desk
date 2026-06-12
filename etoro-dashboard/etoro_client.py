import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# One-time guard: log raw position keys once if no open-date field is recognised.
_LOGGED_POSITION_KEYS = False

_shared_clients: dict[tuple[str, str], "EToroClient"] = {}
_clients_lock = threading.Lock()

# Instrument metadata never changes during a session — cache it forever.
# Key: instrument_id (int)  Value: {"name": str, "symbol": str}
_instrument_meta_cache: dict[int, dict] = {}
_meta_cache_lock = threading.Lock()

# Market-open verdicts (shared across all client instances so ~60 bot threads
# cost at most one rates call per instrument per MARKET_OPEN_TTL).
# Key: instrument_id  Value: (checked_at_monotonic, is_open)
_market_open_cache: dict[int, tuple] = {}
_market_open_lock = threading.Lock()

# Shared historical-candle cache.  Historical OHLC for a given (instrument,
# interval) is IDENTICAL for every bot, so we fetch it once and share it rather
# than letting ~30 bots each hit eToro on boot (which caused read-timeout
# bursts).  The per-key lock de-duplicates the simultaneous boot stampede so only
# ONE network call is made per stream; a short TTL keeps the seed candles fresh.
# Value: (fetched_at_monotonic, DataFrame, count).
_hist_cache: dict[tuple[int, int], tuple] = {}
_hist_cache_lock = threading.Lock()                            # guards _hist_cache + _hist_key_locks
_hist_key_locks: dict[tuple[int, int], threading.Lock] = {}
_HIST_CACHE_TTL = 90.0   # seconds before a stream's history is refetched


def get_shared_client(api_key: str, user_key: str) -> "EToroClient":
    """One HTTP session pool per credential pair (engine + UI)."""
    key = (api_key, user_key)
    with _clients_lock:
        if key not in _shared_clients:
            _shared_clients[key] = EToroClient(api_key, user_key)
        return _shared_clients[key]


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    # Pool sized for the whole bot fleet sharing ONE session: ~86 engine threads
    # plus the positions poller and UI can have dozens of requests in flight at
    # candle close.  pool_maxsize=8 caused constant "pool is full, discarding
    # connection" churn (TLS re-handshake per call) once the fleet grew.
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=48)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class EToroClient:
    BASE_URL = "https://public-api.etoro.com/api/v1"

    def __init__(self, api_key: str, user_key: str):
        self.api_key  = api_key
        self.user_key = user_key
        self._session = _build_session()

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key":     self.api_key,
            "x-user-key":    self.user_key,
            "x-request-id":  str(uuid.uuid4()),
            "Accept":         "application/json",
            "Content-Type":   "application/json",
        }

    def _get(self, endpoint: str, params=None, timeout: int = 15) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                resp = self._session.get(
                    url, headers=self._headers(), params=params, timeout=timeout
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning("Rate limited — sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 403:
                    raise PermissionError(
                        f"403 Forbidden — your API key does not have permission for this endpoint. "
                        f"Enable the required scope at eToro → Settings → API."
                    )
                resp.raise_for_status()
                return resp.json()
            except PermissionError:
                raise
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise TimeoutError(f"Request timed out after {timeout}s: {endpoint}")
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError as exc:
                if attempt == 2:
                    raise ConnectionError(f"Connection failed: {exc}") from exc
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(f"HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        raise RuntimeError(f"All retries exhausted for {endpoint}")

    def _post(self, endpoint: str, body: dict, timeout: int = 30) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                resp = self._session.post(
                    url, headers=self._headers(), json=body, timeout=timeout
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 403:
                    raise PermissionError(
                        "403 Forbidden — enable Trading scope on your eToro API key."
                    )
                resp.raise_for_status()
                return resp.json() if resp.content else {}
            except PermissionError:
                raise
            except requests.exceptions.HTTPError as exc:
                detail = exc.response.text[:300] if exc.response is not None else str(exc)
                raise RuntimeError(f"HTTP {exc.response.status_code}: {detail}") from exc
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"All retries exhausted for POST {endpoint}")

    # ── Market Data ──────────────────────────────────────────────────────────

    def search_instruments(self, query: str) -> Any:
        return self._get("/market-data/search", params={"query": query})

    def get_instruments(self, instrument_ids: Optional[List[int]] = None) -> Any:
        if not instrument_ids:
            return self._get("/market-data/instruments", timeout=30)
        results: List[Dict] = []
        def fetch_one(iid):
            try:
                return self._get("/market-data/instruments", params=[("instrumentIds", iid)])
            except Exception:
                return {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_one, iid): iid for iid in instrument_ids}
            for f in as_completed(futures):
                data = f.result()
                results.extend(data.get("instrumentDisplayDatas", []))
        return {"instrumentDisplayDatas": results}

    def get_rates(self, instrument_ids: Optional[List[int]] = None) -> Any:
        params = [("instrumentIds", iid) for iid in instrument_ids] if instrument_ids else None
        return self._get("/market-data/instruments/rates", params=params)

    # ── Market-hours awareness ────────────────────────────────────────────────
    # eToro's public API exposes no explicit open/closed flag, so this layers
    # two signals (most-authoritative first):
    #   1. SESSION CALENDAR (stocks/ETFs) — market_calendar computes the US
    #      equity session by RULE: weekends, every NYSE/NASDAQ holiday (incl.
    #      Good Friday, observed-date shifts), and 13:00 ET half-days.  This is
    #      PROACTIVE — a holiday is known before a single order is attempted.
    #   2. RATE FRESHNESS — an instrument's rate timestamp freezes when its
    #      exchange stops trading (crypto runs 24/7 so it is always fresh).
    #      Catches what no calendar can know: halts, ad-hoc closures, and
    #      non-US listings.
    # Crypto short-circuits to True with no REST call at all.

    MARKET_OPEN_TTL = 60.0    # seconds a verdict is cached per instrument
    RATE_FRESH_SEC  = 300.0   # rate timestamp older than this ⇒ market closed

    @staticmethod
    def _parse_rate_date(s: str) -> Optional[datetime]:
        """Parse eToro's rate date (e.g. '2026-06-10T13:46:18.9303067Z') — the
        7-digit fraction exceeds fromisoformat's tolerance, so trim to 6."""
        if not s:
            return None
        try:
            s = re.sub(r"\.(\d{6})\d+", r".\1", str(s).strip()).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _rate_is_fresh(self, instrument_id: int) -> bool:
        """Fallback signal: True when the live rate timestamp is recent.
        FAILS OPEN on API errors / unparseable dates."""
        try:
            rates = (self.get_rates([instrument_id]) or {}).get("rates") or []
            if rates:
                dt = self._parse_rate_date(str(rates[0].get("date") or ""))
                if dt is not None:
                    age = (datetime.now(tz=timezone.utc) - dt).total_seconds()
                    return age < self.RATE_FRESH_SEC
        except Exception as exc:
            log.debug("rate freshness check failed for %s (fail-open): %s",
                      instrument_id, exc)
        return True

    def is_market_open(self, instrument_id: int) -> bool:
        """True when the instrument is currently tradable (its market is open).

        Stocks/ETFs: US session calendar first (holidays/weekends/half-days
        known in advance), then rate freshness as live confirmation while the
        calendar says open.  Crypto: always True (24/7, zero REST).  Other
        classes (forex/commodity/index/unknown): rate freshness only."""
        now = time.monotonic()
        with _market_open_lock:
            hit = _market_open_cache.get(instrument_id)
            if hit and now - hit[0] < self.MARKET_OPEN_TTL:
                return hit[1]

        klass = self.asset_class_for(instrument_id)
        if klass == "crypto":
            verdict = True              # 24/7 — no calendar, no REST needed
        elif klass in ("stock", "etf"):
            import market_calendar
            if not market_calendar.is_us_equity_open():
                # NOTE: applies the US calendar to all eToro stocks/ETFs (our
                # fleet is US-listed).  A non-US listing would at worst pause
                # on a US holiday — the conservative direction.
                verdict = False
            else:
                verdict = self._rate_is_fresh(instrument_id)
        else:
            verdict = self._rate_is_fresh(instrument_id)

        with _market_open_lock:
            _market_open_cache[instrument_id] = (now, verdict)
        return verdict

    def get_candles(
        self,
        instrument_id: int,
        direction: str = "desc",
        interval: str = "OneMinute",
        count: int = 200,
    ) -> Any:
        return self._get(
            f"/market-data/instruments/{instrument_id}/history/candles"
            f"/{direction}/{interval}/{count}"
        )

    # interval_seconds → eToro API interval name
    _INTERVAL_NAMES: dict[int, str] = {
        60:    "OneMinute",
        300:   "FiveMinutes",
        600:   "TenMinutes",
        900:   "FifteenMinutes",
        1800:  "ThirtyMinutes",
        3600:  "OneHour",
        14400: "FourHours",
        86400: "OneDay",
    }

    def get_hist_candles(
        self,
        instrument_id: int,
        interval_seconds: int,
        count: int = 200,
    ):
        """
        Fetch historical OHLC candles and return a clean pandas DataFrame with
        columns [time, Open, High, Low, Close].

        Returns an empty DataFrame if the API call fails or returns no rows.
        """
        import pandas as pd

        api_name = self._INTERVAL_NAMES.get(interval_seconds, "OneMinute")
        try:
            raw = self.get_candles(instrument_id, "desc", api_name, count)
        except Exception as exc:
            log.warning("get_hist_candles fetch failed for %s: %s", instrument_id, exc)
            return pd.DataFrame()

        # Navigate the nested response: {candles: [{candles: [...]}]}
        outer = []
        if isinstance(raw, list):
            outer = raw
        elif isinstance(raw, dict):
            for key in ("candles", "data", "Candles"):
                val = raw.get(key)
                if isinstance(val, list):
                    outer = val
                    break

        rows: list = []
        if outer:
            first = outer[0] if outer else {}
            if isinstance(first, dict):
                rows = first.get("candles", first.get("data", []))
            elif isinstance(first, list):
                rows = first
        if not rows and isinstance(raw, list):
            rows = raw

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        rename: dict[str, str] = {}
        for col in df.columns:
            lc = col.lower()
            if lc in ("fromdate", "date", "time", "timestamp", "datetime"):
                rename[col] = "time"
            elif lc == "open":    rename[col] = "Open"
            elif lc == "high":    rename[col] = "High"
            elif lc == "low":     rename[col] = "Low"
            elif lc == "close":   rename[col] = "Close"
        df = df.rename(columns=rename)

        needed = {"time", "Open", "High", "Low", "Close"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"])
        for c in ("Open", "High", "Low", "Close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("time").reset_index(drop=True)
        return df[["time", "Open", "High", "Low", "Close"]]

    def get_hist_candles_cached(
        self,
        instrument_id: int,
        interval_seconds: int,
        count: int = 200,
        ttl: float = _HIST_CACHE_TTL,
    ):
        """Shared, de-duplicated wrapper over get_hist_candles.

        All bots on the same (instrument, interval) consume identical history, so
        the first caller fetches and every concurrent / later caller reuses the
        cached DataFrame until it goes stale (ttl).  This collapses the ~30 boot
        history requests down to one per stream and stops the read-timeout bursts.
        """
        import pandas as pd

        key = (int(instrument_id), int(interval_seconds))

        def _fresh_hit() -> "Optional[Any]":
            entry = _hist_cache.get(key)
            if (
                entry is not None
                and (time.monotonic() - entry[0]) < ttl
                and entry[2] >= count
                and not entry[1].empty
            ):
                return entry[1].copy()
            return None

        # Fast path — fresh cache hit without serialising on the per-key lock.
        with _hist_cache_lock:
            hit = _fresh_hit()
            if hit is not None:
                return hit
            key_lock = _hist_key_locks.get(key)
            if key_lock is None:
                key_lock = threading.Lock()
                _hist_key_locks[key] = key_lock

        # Serialise fetches for this stream so a boot stampede makes ONE call.
        with key_lock:
            # Re-check: another thread may have populated it while we waited.
            with _hist_cache_lock:
                hit = _fresh_hit()
                if hit is not None:
                    return hit

            df = self.get_hist_candles(instrument_id, interval_seconds, count)

            if df is not None and not df.empty:
                with _hist_cache_lock:
                    _hist_cache[key] = (time.monotonic(), df, int(count))
                return df.copy()

            # Fetch failed/empty — serve a stale copy if we have one, rather than
            # returning nothing (a transient timeout shouldn't blank the chart).
            with _hist_cache_lock:
                entry = _hist_cache.get(key)
                if entry is not None and not entry[1].empty:
                    return entry[1].copy()
            return df if df is not None else pd.DataFrame()

    def get_instrument_types(self) -> Any:
        return self._get("/market-data/instrument-types")

    def get_exchanges(self) -> Any:
        return self._get("/market-data/exchanges")

    # ── Portfolio & Trading ───────────────────────────────────────────────────

    def get_portfolio(self, demo: bool = False) -> Any:
        path = "/trading/info/demo/portfolio" if demo else "/trading/info/portfolio"
        return self._get(path)

    def get_pnl(self, demo: bool = False) -> Any:
        path = "/trading/info/demo/pnl" if demo else "/trading/info/real/pnl"
        return self._get(path)

    def get_trade_history(
        self, min_date: str, page: int = 1, page_size: int = 50, *, demo: bool = False
    ) -> Any:
        path = (
            "/trading/info/trade/demo/history"
            if demo
            else "/trading/info/trade/history"
        )
        return self._get(
            path,
            params={"minDate": min_date, "pageNumber": page, "pageSize": page_size},
        )

    def get_all_trade_history(
        self,
        min_date: str,
        *,
        demo: bool = False,
        page_size: int = 200,
        max_pages: int = 25,
    ) -> list[dict]:
        """Fetch every page and dedupe by position id (demo API may repeat pages)."""
        seen: set[int | str] = set()
        all_trades: list[dict] = []
        for page in range(1, max_pages + 1):
            raw = self.get_trade_history(
                min_date, page=page, page_size=page_size, demo=demo,
            )
            batch = self._normalize_history_batch(raw)
            if not batch:
                break
            added = 0
            for trade in batch:
                pid = trade.get("positionId") or trade.get("position_id")
                key: int | str = pid if pid is not None else f"{page}:{len(all_trades)}"
                if key in seen:
                    continue
                seen.add(key)
                all_trades.append(trade)
                added += 1
            if added == 0 or len(batch) < page_size:
                break
        return all_trades

    @staticmethod
    def _normalize_history_batch(raw: Any) -> list[dict]:
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            for key in (
                "trades", "closedPositions", "history",
                "ClosedPositions", "data", "History",
            ):
                val = raw.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
        return []

    # ── Demo execution (virtual paper money) ─────────────────────────────────

    def open_demo_market_by_amount(
        self,
        instrument_id: int,
        is_buy: bool,
        amount: float,
        leverage: int = 1,
        stop_loss_rate: Optional[float] = None,
    ) -> Any:
        body: dict[str, Any] = {
            "InstrumentID": instrument_id,
            "IsBuy": is_buy,
            "Leverage": leverage,
            "Amount": amount,
            "IsNoTakeProfit": True,
        }
        if stop_loss_rate is not None:
            body["StopLossRate"] = stop_loss_rate
            body["IsNoStopLoss"] = False
        else:
            body["IsNoStopLoss"] = True

        return self._post(
            "/trading/execution/demo/market-open-orders/by-amount",
            body,
        )

    def close_demo_position(
        self,
        position_id: int,
        instrument_id: int,
        units_to_deduct: Optional[float] = None,
    ) -> Any:
        body: dict[str, Any] = {"InstrumentID": instrument_id}
        if units_to_deduct is not None:
            body["UnitsToDeduct"] = units_to_deduct
        return self._post(
            f"/trading/execution/demo/market-close-orders/positions/{position_id}",
            body,
        )

    @staticmethod
    def _dig_positions(data: Any) -> list[dict]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if not isinstance(data, dict):
            return []
        for key in ("positions", "Positions", "openPositions", "clientPortfolio", "data"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                nested = EToroClient._dig_positions(val)
                if nested:
                    return nested
        return []

    @staticmethod
    def _position_id_from_obj(obj: dict) -> Optional[int]:
        for key in ("positionID", "positionId", "PositionID", "position_id", "id"):
            val = obj.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _instrument_id_from_obj(obj: dict) -> Optional[int]:
        for key in ("instrumentID", "InstrumentID", "instrumentId"):
            val = obj.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _is_buy_from_obj(obj: dict) -> Optional[bool]:
        for key in ("isBuy", "IsBuy", "isSettled", "IsSettled"):
            if key in obj:
                return bool(obj[key])
        for key in ("direction", "Direction", "positionType"):
            val = str(obj.get(key, "")).lower()
            if val in ("buy", "long", "1"):
                return True
            if val in ("sell", "short", "0"):
                return False
        return None

    def extract_position_id(self, response: Any) -> Optional[int]:
        if not isinstance(response, dict):
            return None
        pid = self._position_id_from_obj(response)
        if pid:
            return pid
        for key in ("orderForOpen", "order", "data", "position"):
            sub = response.get(key)
            if isinstance(sub, dict):
                pid = self._position_id_from_obj(sub)
                if pid:
                    return pid
        return None

    @staticmethod
    def summarize_order_response(response: Any, *, max_len: int = 500) -> str:
        """Compact one-line summary of an eToro open-order payload for logs/UI."""
        if response is None:
            return "(empty response)"
        if not isinstance(response, dict):
            return str(response)[:max_len]
        parts: list[str] = []
        for key in (
            "ErrorCode", "errorCode", "error", "Error", "message", "Message",
            "status", "Status", "orderID", "OrderID", "token", "Token",
        ):
            val = response.get(key)
            if val not in (None, "", {}):
                parts.append(f"{key}={val}")
        pid = EToroClient._position_id_from_obj(response)
        if pid:
            parts.append(f"position_id={pid}")
        for nest in ("orderForOpen", "order", "data", "position", "clientPortfolio"):
            sub = response.get(nest)
            if not isinstance(sub, dict):
                continue
            for key in (
                "ErrorCode", "errorCode", "message", "Message",
                "orderID", "OrderID", "positionID", "positionId", "id",
            ):
                val = sub.get(key)
                if val not in (None, "", {}):
                    parts.append(f"{nest}.{key}={val}")
        if parts:
            return "; ".join(str(p) for p in parts)[:max_len]
        try:
            import json
            return json.dumps(response, default=str)[:max_len]
        except Exception:
            return str(response)[:max_len]

    def find_demo_position(
        self, instrument_id: int, is_buy: bool
    ) -> Optional[dict]:
        raw = self.get_portfolio(demo=True)
        for pos in self._dig_positions(raw):
            iid = self._instrument_id_from_obj(pos)
            if iid != instrument_id:
                continue
            pos_buy = self._is_buy_from_obj(pos)
            if pos_buy is not None and pos_buy != is_buy:
                continue
            return pos
        return None

    def position_ids_for_instrument(
        self, instrument_id: int, is_buy: Optional[bool] = None, demo: bool = True
    ) -> set[int]:
        """Set of eToro position IDs currently open for an instrument (optionally
        filtered by direction).  Used to identify the NEW position created by an
        open order via a before/after diff — eToro's by-amount open response does
        not return a position id, and many same-instrument positions can coexist.
        """
        ids: set[int] = set()
        try:
            raw = self.get_portfolio(demo=demo)
        except Exception:
            return ids
        for pos in self._dig_positions(raw):
            if self._instrument_id_from_obj(pos) != instrument_id:
                continue
            if is_buy is not None:
                pos_buy = self._is_buy_from_obj(pos)
                if pos_buy is not None and pos_buy != is_buy:
                    continue
            pid = self._position_id_from_obj(pos)
            if pid is not None:
                ids.add(pid)
        return ids

    def resolve_demo_position_id(
        self,
        open_response: Any,
        instrument_id: int,
        is_buy: bool,
        retries: int = 3,
        delay: float = 1.0,
    ) -> Optional[int]:
        pid = self.extract_position_id(open_response)
        if pid:
            return pid
        for attempt in range(retries):
            time.sleep(delay * (attempt + 1))
            pos = self.find_demo_position(instrument_id, is_buy)
            if pos:
                pid = self._position_id_from_obj(pos)
                if pid:
                    return pid
        return None

    @staticmethod
    def _pick_numeric(obj: dict, *keys: str) -> Optional[float]:
        for key in keys:
            val = obj.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _pick_str(obj: dict, *keys: str) -> str:
        for key in keys:
            val = obj.get(key)
            if val:
                return str(val)
        return ""

    def normalize_position(self, pos: dict) -> dict:
        is_buy = self._is_buy_from_obj(pos)
        if is_buy is True:
            direction = "LONG"
        elif is_buy is False:
            direction = "SHORT"
        else:
            direction = "—"

        normalized = {
            "position_id":    self._position_id_from_obj(pos),
            "instrument_id": self._instrument_id_from_obj(pos),
            "is_buy":         is_buy,
            "direction":      direction,
            "name": self._pick_str(
                pos,
                "instrumentDisplayName", "InstrumentDisplayName",
                "symbolFull", "SymbolFull", "name", "Name", "instrumentName",
            ),
            "symbol": self._pick_str(pos, "symbolFull", "SymbolFull", "symbol", "Symbol"),
            "open_rate": self._pick_numeric(
                pos, "openRate", "OpenRate", "openPrice", "OpenPrice",
                "averageOpen", "AverageOpen", "avgOpen", "AvgOpen",
            ),
            "pnl": self._pick_numeric(
                pos, "profit", "Profit", "pnl", "PnL",
                "unrealizedPnL", "UnrealizedPnL", "grossProfit", "GrossProfit",
            ),
            "amount": self._pick_numeric(
                pos, "amount", "Amount", "initialAmountInDollars",
                "InitialAmountInDollars", "investment", "Investment",
            ),
            "units": self._pick_numeric(
                pos, "units", "Units", "amountInUnits", "AmountInUnits",
            ),
            "stop_loss": self._pick_numeric(
                pos, "stopLossRate", "StopLossRate", "stopLoss", "StopLoss",
            ),
            "leverage": self._pick_numeric(pos, "leverage", "Leverage"),
            # Open timestamp — used to place the entry arrow on the correct candle.
            # eToro uses various camelCase / PascalCase field names (and sometimes
            # an epoch) depending on the API version / endpoint; try all variants.
            "open_date": self._pick_str(
                pos,
                "openDateTime", "OpenDateTime",
                "openDate", "OpenDate",
                "openedDate", "OpenedDate",
                "openDateUtc", "OpenDateUtc",
                "openTime", "OpenTime",
                "openTimestamp", "OpenTimestamp",
                "creationDate", "CreationDate",
                "createDate", "CreateDate",
                "dateOpened", "DateOpened",
                "timestamp", "Timestamp",
            ),
            "current_rate": None,
            "current_value": None,
            "pnl_pct": None,
            "status": "—",
        }
        # One-time observability: if we couldn't find an open timestamp, log the
        # raw keys ONCE so the actual eToro field name can be identified and added
        # above.  Without this the entry-arrow falls back to a price-based guess.
        global _LOGGED_POSITION_KEYS
        if not normalized["open_date"] and not _LOGGED_POSITION_KEYS:
            _LOGGED_POSITION_KEYS = True
            log.warning(
                "eToro position has no recognised open-date field. Available keys: %s",
                sorted(pos.keys()),
            )
        return normalized

    def _fetch_instrument_meta(self, instrument_ids: list[int]) -> dict[int, dict]:
        if not instrument_ids:
            return {}

        # Return fully from session-level cache when all IDs are already known.
        with _meta_cache_lock:
            missing = [iid for iid in instrument_ids if iid not in _instrument_meta_cache]
            if not missing:
                return {iid: _instrument_meta_cache[iid] for iid in instrument_ids}

        try:
            raw = self.get_instruments(missing)
        except Exception as exc:
            log.warning("Instrument lookup failed: %s", exc)
            with _meta_cache_lock:
                return {iid: _instrument_meta_cache.get(iid, {"name": "", "symbol": ""})
                        for iid in instrument_ids}

        with _meta_cache_lock:
            for inst in raw.get("instrumentDisplayDatas", []):
                iid = inst.get("instrumentID")
                if iid is None:
                    continue
                _instrument_meta_cache[int(iid)] = {
                    "name":        inst.get("instrumentDisplayName", ""),
                    "symbol":      inst.get("symbolFull", ""),
                    "type_id":     inst.get("instrumentTypeID"),
                    "exchange_id": inst.get("exchangeID"),
                }
            return {iid: _instrument_meta_cache.get(iid, {"name": "", "symbol": ""})
                    for iid in instrument_ids}

    # eToro instrumentTypeID → our asset-class taxonomy (see /market-data/
    # instrument-types):  1 Forex · 2 Commodity · 3 CFD · 4 Indices · 5 Stocks ·
    # 6 ETF · 7 Bonds · 8 TrustFunds · 9 Options · 10 Crypto
    _TYPE_ID_TO_CLASS = {
        1:  "forex",
        2:  "commodity",
        4:  "index",
        5:  "stock",
        6:  "etf",
        10: "crypto",
    }

    def asset_class_for(self, instrument_id: int) -> str:
        """Authoritative asset class from eToro instrument metadata.

        Returns '' when the type can't be determined (caller should fall back
        to label-keyword heuristics)."""
        try:
            meta = self._fetch_instrument_meta([int(instrument_id)])
            type_id = (meta.get(int(instrument_id)) or {}).get("type_id")
            return self._TYPE_ID_TO_CLASS.get(int(type_id), "") if type_id is not None else ""
        except Exception as exc:
            log.debug("asset_class_for(%s) failed: %s", instrument_id, exc)
            return ""

    def _fetch_rates_map(self, instrument_ids: list[int]) -> dict[int, dict]:
        """Fetch live bid/ask for each position instrument.

        eToro's /rates endpoint ignores extra instrumentIds query params and
        returns only one row per request — parallel single-id fetches required.
        """
        rates: dict[int, dict] = {}
        if not instrument_ids:
            return rates

        def fetch_one(iid: int) -> tuple[int, dict] | None:
            try:
                data = self.get_rates([iid])
                for row in data.get("rates", []):
                    rid = row.get("instrumentID")
                    if rid is not None:
                        return int(rid), row
            except Exception as exc:
                log.warning("Rate fetch failed for %s: %s", iid, exc)
            return None

        workers = min(6, len(instrument_ids))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(fetch_one, instrument_ids):
                if result:
                    rates[result[0]] = result[1]
        return rates

    @staticmethod
    def _compute_live_pnl(pos: dict, rate: dict) -> dict:
        """Fill current price, value, P&L from live bid/ask when API omits them."""
        bid = rate.get("bid")
        ask = rate.get("ask")
        units = pos.get("units") or 0
        amount = pos.get("amount") or 0
        is_buy = pos.get("is_buy")

        if is_buy is True and bid and units:
            current_value = units * float(bid)
            pnl = current_value - amount
            pos["current_rate"] = float(bid)
            pos["current_value"] = current_value
        elif is_buy is False and ask and units:
            current_value = units * float(ask)
            pnl = amount - current_value
            pos["current_rate"] = float(ask)
            pos["current_value"] = current_value
        else:
            return pos

        if pos.get("pnl") is None:
            pos["pnl"] = pnl
        if amount:
            pos["pnl_pct"] = (pos["pnl"] / amount) * 100

        ch = EToroClient._pick_numeric(
            rate,
            "dailyPriceChange", "DailyPriceChange",
            "priceChange", "PriceChange", "change", "Change",
        )
        ch_pct = EToroClient._pick_numeric(
            rate,
            "dailyPriceChangeInPercent", "DailyPriceChangeInPercent",
            "priceChangeInPercent", "PriceChangeInPercent",
            "changePercent", "ChangePercent",
        )
        if ch is not None:
            pos["daily_change"] = ch
        if ch_pct is not None:
            pos["daily_change_pct"] = ch_pct

        if pos["pnl"] > 0.005:
            pos["status"] = "Gaining"
        elif pos["pnl"] < -0.005:
            pos["status"] = "Losing"
        else:
            pos["status"] = "Flat"

        return pos

    def enrich_positions(self, positions: list[dict]) -> list[dict]:
        ids = list({p["instrument_id"] for p in positions if p.get("instrument_id")})
        if not ids:
            return positions

        meta  = self._fetch_instrument_meta(ids)
        rates = self._fetch_rates_map(ids)

        enriched = []
        for pos in positions:
            p = dict(pos)
            iid = p.get("instrument_id")
            if iid in meta:
                p["name"]   = p["name"]   or meta[iid]["name"]
                p["symbol"] = p["symbol"] or meta[iid]["symbol"]
            if iid in rates:
                p = self._compute_live_pnl(p, rates[iid])
            enriched.append(p)
        return enriched

    def get_open_positions(self, demo: bool = False) -> list[dict]:
        raw = self.get_portfolio(demo=demo)
        dug = self._dig_positions(raw)
        positions = [self.normalize_position(p) for p in dug]
        # Keep only GENUINE open positions.  Some portfolio responses also list
        # pending orders or zero-size / settled entries; a real open position has
        # a resolvable position id AND a nonzero size (units or amount).
        positions = [
            p for p in positions
            if p.get("position_id") is not None
            and ((p.get("units") or 0) > 0 or (p.get("amount") or 0) > 0)
        ]
        if len(positions) != len(dug):
            log.info(
                "get_open_positions(demo=%s): %d raw entries → %d genuine open positions",
                demo, len(dug), len(positions),
            )
        return self.enrich_positions(positions)

    def get_portfolio_raw(self, demo: bool = False) -> Any:
        """Raw portfolio payload exactly as eToro returns it (for the debug view)."""
        return self.get_portfolio(demo=demo)

    def get_position_for_instrument(
        self, instrument_id: int, demo: bool = False
    ) -> Optional[dict]:
        for pos in self.get_open_positions(demo=demo):
            if pos.get("instrument_id") == instrument_id:
                return pos
        return None

    # ── Watchlists ────────────────────────────────────────────────────────────

    def get_watchlists(self) -> Any:
        return self._get("/watchlists")

    def get_watchlist(self, watchlist_id: str) -> Any:
        return self._get(f"/watchlists/{watchlist_id}")

    # ── Social ────────────────────────────────────────────────────────────────

    def get_instrument_feed(self, market_id: str) -> Any:
        return self._get(f"/feeds/instrument/{market_id}")

    def get_user_info(self, username: str) -> Any:
        return self._get("/user-info/people", params={"username": username})

    def get_public_portfolio(self, username: str) -> Any:
        return self._get(f"/user-info/people/{username}/portfolio/live")
