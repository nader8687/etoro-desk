import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session.mount("https://", adapter)
    return session


class EToroClient:
    BASE_URL = "https://public-api.etoro.com/api/v1"

    def __init__(self, api_key: str, user_key: str):
        self.api_key  = api_key
        self.user_key = user_key
        self._session = _build_session()

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key":    self.api_key,
            "x-user-key":   self.user_key,
            "x-request-id": str(uuid.uuid4()),
            "Accept":       "application/json",
        }

    def _get(self, endpoint: str, params=None, timeout: int = 15) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                resp = self._session.get(
                    url, headers=self._headers(), params=params, timeout=timeout
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 403:
                    raise PermissionError("403 — enable Trading scope in eToro API settings.")
                resp.raise_for_status()
                return resp.json()
            except PermissionError:
                raise
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise TimeoutError(f"Timed out: {endpoint}")
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError as exc:
                if attempt == 2:
                    raise ConnectionError(str(exc)) from exc
                time.sleep(2 ** attempt)
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(f"HTTP {exc.response.status_code}") from exc
        raise RuntimeError(f"All retries failed for {endpoint}")

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
            for f in as_completed({pool.submit(fetch_one, iid): iid for iid in instrument_ids}):
                results.extend(f.result().get("instrumentDisplayDatas", []))
        return {"instrumentDisplayDatas": results}

    def get_candles(self, instrument_id: int, direction: str = "desc",
                    interval: str = "OneMinute", count: int = 100) -> Any:
        return self._get(
            f"/market-data/instruments/{instrument_id}/history/candles"
            f"/{direction}/{interval}/{count}"
        )
