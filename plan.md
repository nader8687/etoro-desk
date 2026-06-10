# Multi-Instrument Auto-Trade — Implementation Plan

## Goal

Run independent trading engines for multiple instruments (starting: XRP + BTC) in the
background.  Each instrument has its own WebSocket feed, candle builder, LLM signal loop,
and trade manager state.  Streamlit can view any running instrument at any time and toggle
auto-trade per instrument.  Adding a third instrument later = one new section in a config
file, no code changes.

---

## What Already Supports Multiple Instruments (no changes needed)

| Module | Status | Why |
|---|---|---|
| `tick_manager.py` | ✅ ready | All state is keyed by `instrument_id` — buffers, WS threads, watchdogs |
| `trade_manager.py` | ✅ ready | `_open`, `_closed` keyed by `instrument_id` |
| `signal_worker.py` | ✅ ready | Results keyed by `(instrument_id, interval)` |
| `signal_log.py` | ✅ ready | Single JSONL file, filters by `instrument_id` |
| `engine_notify.py` | ✅ ready | Queue entries carry `instrument_id` |
| `positions_cache.py` | ✅ ready | Returns all positions; engine already filters by `instrument_id` |
| `etoro_client.py` | ✅ ready | Stateless; shared client instance works for all instruments |

## What Must Change

| Module | Change needed |
|---|---|
| `market_data_hub.py` | Single-snapshot globals → `dict[instrument_id → HubState]` |
| `trading_engine.py` | Single-engine globals → `dict[instrument_id → EngineState]` |
| `app.py` | New "Bots" page; read snapshots by `instrument_id` instead of single global |

## New Files

| File | Purpose |
|---|---|
| `instruments.toml` | Declarative config — one section per instrument |
| `instrument_config.py` | Config loader — returns `list[InstrumentSpec]` |

---

## Architecture

```
instruments.toml
       │
       ▼
instrument_config.py  ── InstrumentSpec list ──►  engine_registry (trading_engine.py)
                                                         │
                          ┌──────────────────────────────┼────────────────────────────┐
                          │ instrument: XRP              │ instrument: BTC             │
                          │                              │                             │
                          │  tick_manager[xrp_id]        │  tick_manager[btc_id]       │
                          │       ↓ ticks                │       ↓ ticks               │
                          │  market_data_hub[xrp_id]     │  market_data_hub[btc_id]    │
                          │       ↓ ChartSnapshot        │       ↓ ChartSnapshot       │
                          │  EngineState[xrp_id]         │  EngineState[btc_id]        │
                          │    - _run_tick()             │    - _run_tick()            │
                          │    - signal_worker           │    - signal_worker          │
                          │    - trade_manager           │    - trade_manager          │
                          └──────────────────────────────┴────────────────────────────┘
                                                         │
                                                         ▼
                                               Streamlit app.py
                                               "Bots" page — overview grid
                                               "Trading" tab — pick any running instrument
                                               "Signals" page — already filters by iid
                                               "P&L" tab — already multi-instrument
```

---

## Step-by-Step Implementation

---

### STEP 1 — Config file: `instruments.toml`

**File:** `etoro-dashboard/instruments.toml`

```toml
# Add or remove [instruments.X] sections to control which instruments run.
# Restart the container to pick up changes.
# instrument_id: eToro internal ID (visible in the Trading tab instrument selector).
# auto_trade: false = live feed + LLM signals only; true = engine also opens/closes orders.

[instruments.xrp]
label          = "XRP  (XRP)"
instrument_id  = 2094
interval       = "1 Minute"
interval_secs  = 60
candle_count   = 100
demo_amount    = 100.0
enabled        = true
auto_trade     = false

[instruments.btc]
label          = "Bitcoin  (BTC)"
instrument_id  = 100000
interval       = "1 Minute"
interval_secs  = 60
candle_count   = 100
demo_amount    = 100.0
enabled        = true
auto_trade     = false
```

**Rules:**
- `enabled = false` → instrument is not started at all (no WS, no LLM, no thread).
- `auto_trade = false` (default) → WS feed runs, LLM signals fire, but no orders are placed.
  Streamlit's auto-trade toggle overrides this at runtime without restarting.
- `interval_secs` must match `interval` (e.g. "1 Minute" = 60).
- To add a third instrument: copy any block, change the key + values. No code changes.

---

### STEP 2 — Config loader: `instrument_config.py` (new file)

```python
"""
Loads instruments.toml and exposes InstrumentSpec list.

Usage:
    from instrument_config import load_specs
    specs = load_specs()          # list[InstrumentSpec], enabled only
"""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "instruments.toml"

@dataclass(frozen=True)
class InstrumentSpec:
    key:           str    # "xrp", "btc"
    label:         str    # eToro display label
    instrument_id: int
    interval:      str    # "1 Minute"
    interval_secs: int    # 60
    candle_count:  int    # candles to keep in chart
    demo_amount:   float  # dollars per trade
    enabled:       bool
    auto_trade:    bool   # initial value; runtime state can override

def load_specs(*, enabled_only: bool = True) -> list[InstrumentSpec]:
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    specs = []
    for key, sec in data.get("instruments", {}).items():
        spec = InstrumentSpec(
            key=key,
            label=sec["label"],
            instrument_id=int(sec["instrument_id"]),
            interval=sec["interval"],
            interval_secs=int(sec.get("interval_secs", 60)),
            candle_count=int(sec.get("candle_count", 100)),
            demo_amount=float(sec.get("demo_amount", 100.0)),
            enabled=bool(sec.get("enabled", True)),
            auto_trade=bool(sec.get("auto_trade", False)),
        )
        if enabled_only and not spec.enabled:
            continue
        specs.append(spec)
    return specs
```

**Dependency note:** `tomllib` is part of Python 3.11 stdlib.  If the container image
uses Python 3.10, add `tomli` to `requirements.txt` and import as:
```python
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]
```

---

### STEP 3 — Refactor `market_data_hub.py` → multi-instrument

**Current:** one set of module-level globals (`_config`, `_hist_df`, `_snapshot`, `_thread`, …)
**Target:** `dict[instrument_id, HubState]` with a single shared build loop.

#### 3a. New internal data model

```python
@dataclass
class HubState:
    config:                    HubConfig
    hist_df:                   pd.DataFrame
    snapshot:                  Optional[ChartSnapshot]
    running:                   bool
    thread:                    Optional[threading.Thread]
    last_build_buf_len:        int   = -1
    last_build_instrument_id:  int   = -1
```

Module-level registry:
```python
_hubs:  dict[int, HubState] = {}      # instrument_id → HubState
_lock = threading.Lock()
_build_thread: Optional[threading.Thread] = None
_build_running = False
```

#### 3b. Changed public API (backwards-compatible where possible)

| Old signature | New signature | Notes |
|---|---|---|
| `configure(config, hist_df)` | `configure(config, hist_df)` | creates or updates `_hubs[config.instrument_id]` |
| `set_hist(df)` | `set_hist(instrument_id, df)` | targets one hub |
| `get_snapshot()` | `get_snapshot(instrument_id)` | returns `Optional[ChartSnapshot]` for that iid |
| `get_config()` | `get_config(instrument_id)` | returns `Optional[HubConfig]` |
| `start()` | `start(instrument_id)` | marks hub as running; shared build loop picks it up |
| `stop()` | `stop(instrument_id)` | marks hub as stopped; does NOT stop other hubs |
| `stop_all()` | `stop_all()` | NEW — stops every hub |
| `set_desired_active(bool)` | `set_desired_active(instrument_id, bool)` | per-instrument |

#### 3c. Single shared build loop

Replace the per-instrument thread with one shared loop:

```python
def _build_loop() -> None:
    while _build_running:
        with _lock:
            active = [(iid, s) for iid, s in _hubs.items() if s.running]
        for iid, state in active:
            try:
                _run_build_for(iid, state)
            except Exception:
                log.exception("Hub build failed for instrument %s", iid)
        time.sleep(BUILD_INTERVAL)
```

`_run_build_for(iid, state)` is the existing `_run_build` logic, operating on `state`
instead of module globals.

**Important:** The shared build thread is started once.  Adding a new instrument just
inserts into `_hubs` — no new threads required.

#### 3d. Supervisor change

Current supervisor restarts the single hub thread.  New supervisor checks that the shared
build thread is alive and restarts it if not.  It no longer starts per-instrument threads.

---

### STEP 4 — Refactor `trading_engine.py` → multi-instrument

**Current:** one set of module globals per engine state.
**Target:** per-instrument `EngineState` objects in a registry dict.

#### 4a. New `EngineState` dataclass (replaces the scattered module globals)

```python
@dataclass
class EngineState:
    config:              EngineConfig
    client:              EToroClient
    running:             bool                  = False
    thread:              Optional[threading.Thread] = None
    prev_candle_time:    Optional[pd.Timestamp] = None
    processed_sig_at:    Optional[str]          = None
    processed_exit_at:   Optional[str]          = None
    snapshot:            Optional[EngineSnapshot] = None
    last_positions_poll: float                  = 0.0
    skip_adopt_until:    float                  = 0.0   # monotonic
```

Module-level registry:
```python
_engines: dict[int, EngineState] = {}    # instrument_id → EngineState
_lock = threading.Lock()
_portfolio_bump   = False
_last_closes:     dict[int, ClosedTrade] = {}
_trade_errors:    dict[int, str]         = {}
_desired_live:    bool = True            # global live toggle
```

#### 4b. Key function signatures

| Old | New | Notes |
|---|---|---|
| `ensure_running(config, hist_df)` | `start_instrument(spec, api_key, user_key, hist_df=None)` | creates `EngineState`, starts thread |
| `stop()` | `stop_instrument(instrument_id)` | stops one engine |
| `stop()` (global) | `stop_all()` | stops every engine |
| `get_snapshot()` | `get_snapshot(instrument_id)` | reads from `_engines[iid].snapshot` |
| `get_all_snapshots()` | — | NEW: `dict[int, EngineSnapshot]` for Bots page |
| `set_desired_trading_active(bool)` | `set_auto_trade(instrument_id, bool)` | per-instrument flag |
| `is_desired_trading_active()` | `is_auto_trade(instrument_id)` | per-instrument |
| `set_desired_live(bool)` | `set_live(instrument_id, bool)` OR keep global | keep global is fine |
| `update_from_ui(config, hist_df)` | `update_instrument_from_ui(spec, hist_df)` | UI pushes to one instrument |

#### 4c. Per-instrument tick loop

The existing `_run_tick()` function becomes `_run_tick(iid, state, config, client)`:
- All global state reads (`_running`, `_config`, `_prev_candle_time`, etc.) become
  `state.running`, `state.config`, `state.prev_candle_time`, etc.
- Each `EngineState.thread` runs a simple loop:
  ```python
  def _instrument_loop(iid: int) -> None:
      while True:
          with _lock:
              state = _engines.get(iid)
              if state is None or not state.running:
                  break
          try:
              _run_tick(iid, state, state.config, state.client)
          except Exception:
              log.exception("Engine tick failed for instrument %s", iid)
          time.sleep(TICK_INTERVAL)
  ```

#### 4d. Supervisor (single shared thread)

```python
def _supervisor_loop() -> None:
    while True:
        try:
            with _lock:
                items = list(_engines.items())
            for iid, state in items:
                tick_manager.start(iid, state.config.api_key, state.config.user_key)
                if state.running and (state.thread is None or not state.thread.is_alive()):
                    _restart_engine_thread(iid, state)
        except Exception:
            log.exception("Engine supervisor tick failed")
        time.sleep(3)
```

One supervisor, N engines.  No per-instrument supervisors.

#### 4e. Historical data preload (background)

`start_instrument` spawns a one-shot thread to load hist candles and feed them to the hub:

```python
def _preload_hist(iid: int, spec: InstrumentSpec, client: EToroClient) -> None:
    try:
        df = client.get_hist_candles(iid, spec.interval_secs, spec.candle_count + 20)
        market_data_hub.set_hist(iid, df)
        log.info("Hist loaded for %s (%d candles)", spec.label, len(df))
    except Exception as exc:
        log.warning("Hist preload failed for %s: %s", spec.label, exc)
```

The engine starts ticking immediately on live data; hist backfills transparently within ~5 s.

---

### STEP 5 — Startup: `app.py` boot sequence

Currently `app.py` calls `trading_engine.ensure_running(config)` only when a user
visits the Trading tab.  With multi-instrument, the engines must start at module load time
(not on first user interaction).

Add to top of `app.py` (runs once per Python process, survives Streamlit reruns):

```python
import instrument_config
import trading_engine

def _boot_background_engines() -> None:
    """Start one engine per enabled instrument on first load."""
    if st.session_state.get("_engines_booted"):
        return
    st.session_state["_engines_booted"] = True

    specs = instrument_config.load_specs()
    api_key  = os.environ["ETORO_API_KEY"]
    user_key = os.environ["ETORO_USER_KEY"]
    for spec in specs:
        trading_engine.start_instrument(spec, api_key, user_key)
        log.info("Background engine started: %s", spec.label)

_boot_background_engines()
```

**Note:** `st.session_state["_engines_booted"]` persists within a browser session.
The module-level engine registry (`_engines` dict in trading_engine.py) persists for
the lifetime of the Python process, so engines survive Streamlit reruns and tab
switches.  The boot guard just prevents duplicate `start_instrument` calls on the
same engine after a page rerun.

---

### STEP 6 — New "Bots" page in Streamlit

Add `"Bots"` to `_NAV_OPTIONS` (after "Trading", before "Portfolio"):

```python
_NAV_OPTIONS = ["Trading", "Bots", "Portfolio", "History", "P&L", "Signals", "Watchlists"]
```

#### 6a. Layout — overview grid

Each running instrument renders as a row with these columns:

```
| Instrument       | WS    | Position         | Last Signal       | Auto-trade | |
| Bitcoin (BTC)    | 🟢 Live | SHORT @ 68,420  | SELL_SHORT 87%   |  [toggle]  | [View] |
| XRP (XRP)        | 🟢 Live | –               | HOLD 60%         |  [toggle]  | [View] |
```

Fragment: `@st.fragment(run_every=5)` so it refreshes every 5 s.

Data sources per row:
- WS status: `tick_manager.get_state(iid)` + `tick_manager.get_last_tick_time(iid)`
- Position: `trade_manager.get_open(iid)` or `positions_cache` filtered by iid
- Last signal: `signal_worker.get_result(iid, interval)` + `signal_worker.get_exit_result(iid, interval)`
- Auto-trade: `trading_engine.is_auto_trade(iid)` → toggle calls `trading_engine.set_auto_trade(iid, bool)`

#### 6b. Auto-trade toggle

```python
current = trading_engine.is_auto_trade(spec.instrument_id)
toggled = st.toggle(
    "Auto-trade",
    value=current,
    key=f"bot_autotrade_{spec.instrument_id}",
)
if toggled != current:
    trading_engine.set_auto_trade(spec.instrument_id, toggled)
```

This is the Streamlit control for each instrument — does NOT affect the other instruments.

#### 6c. "View" button → jump to Trading tab

```python
if st.button("View", key=f"bot_view_{spec.instrument_id}"):
    st.session_state["engine_selected_label"] = spec.label
    st.session_state["main_nav"] = "Trading"
    st.rerun()
```

This sets the active instrument in the Trading tab and navigates there.

#### 6d. P&L summary row per instrument

Below the grid, a small expandable showing:
```
Bitcoin (BTC) — Session P&L: +$4.20 | 2 closed trades | 1 open
```

Sourced from `trade_manager.total_realised_pnl(iid)` and `trade_manager.get_closed(iid)`.

---

### STEP 7 — Trading tab adaptation

The Trading tab currently drives the single engine via `trading_engine.update_from_ui(config)`.
In multi-instrument mode, the Trading tab still works the same way — the user picks one
instrument from the selector, and the tab pushes config to that specific engine.

Changes needed in Trading tab:
1. `trading_engine.get_snapshot()` → `trading_engine.get_snapshot(selected_instrument_id)`
2. `market_data_hub.get_snapshot()` → `market_data_hub.get_snapshot(selected_instrument_id)`
3. `trading_engine.update_from_ui(config, hist_df)` → `trading_engine.update_instrument_from_ui(spec, hist_df)` (no conceptual change, just different function name)

The auto-trade toggle in the Trading tab now also syncs via `set_auto_trade(iid, bool)`.
It only affects the selected instrument — the others keep running.

---

### STEP 8 — Signals page (already ready)

No changes.  The Signals page already:
- Filters by `instrument_id` (multiselect from `ALL_INSTRUMENTS`)
- Uses `signal_log.load(instrument_ids=[...])` which is instrument-keyed
- Auto-refreshes every 30 s via fragment

Once BTC and XRP engines are running and logging signals, the Signals page will show
both streams without modification.

---

### STEP 9 — P&L tab (minor adaptation)

Currently shows all closed trades from `trade_manager.get_closed()`.  This already
includes all instruments.  Add an optional instrument filter:

```python
pnl_instruments = list({t.instrument_label for t in all_closed})
if len(pnl_instruments) > 1:
    sel = st.multiselect("Filter by instrument", pnl_instruments, default=pnl_instruments)
    display_trades = [t for t in all_closed if t.instrument_label in sel]
```

---

### STEP 10 — Dockerfile / docker-compose

Add `instruments.toml` to the Dockerfile copy:

```dockerfile
COPY instruments.toml .
```

No other changes needed.  The named volume (`etoro-data`) already persists the runtime
state and signal log.  `instruments.toml` is baked into the image and versioned in git —
that is the right place for it since it defines the trading configuration.

To override for a specific deployment without rebuilding, mount a file:
```yaml
volumes:
  - ./instruments_override.toml:/app/instruments.toml:ro
```

---

## Threading Model Summary

After the refactor, these threads run per-instrument:

| Thread | Count | Purpose |
|---|---|---|
| `ws-{iid}` (tick_manager) | 1 per instrument | WebSocket feed |
| `wd-{iid}` (tick_manager) | 1 per instrument | WS watchdog / reconnect |
| `seed-{iid}` (tick_manager) | 1 per instrument (short-lived) | REST quote seed on connect |
| `hist-{iid}` (new) | 1 per instrument (short-lived) | Preload historical candles |
| `engine-{iid}` (trading_engine) | 1 per instrument | Trading loop |
| `sig-{iid}:{interval}` (signal_worker) | 1 per LLM call (short-lived) | Fire-and-forget LLM request |

These threads are shared (one regardless of instrument count):

| Thread | Purpose |
|---|---|
| `hub-build` (market_data_hub) | Candle builder — iterates all active hubs |
| `hub-supervisor` (market_data_hub) | Restarts hub-build if dead |
| `engine-supervisor` (trading_engine) | Restarts dead engine threads |

**For 2 instruments (XRP + BTC):** ≈ 10 long-lived threads total.  Very low overhead.
**For 10 instruments:** ≈ 30 long-lived threads.  Still trivially within Python's limits.

---

## Scaling Rules

To add instrument N:

1. Find the eToro `instrument_id` from the Trading tab instrument selector (it's logged at startup).
2. Add a `[instruments.key]` block to `instruments.toml`.
3. Rebuild: `docker compose up --build -d etoro-dashboard`.
4. Done.  A new engine thread starts automatically.

No code changes are ever needed to add an instrument.

Practical limits per single container:
- **~20 instruments** before WS reconnect storms and REST rate-limiting become a concern.
- At 20 instruments, consider batching WS subscriptions (one WS per N instruments instead
  of one per instrument) — eToro's Subscribe message already accepts a list of topics:
  `"topics": ["instrument:100000", "instrument:2094"]`.  This is a future optimisation
  and not needed for 2–10 instruments.

---

## Testing Checklist

After implementation, verify:

- [ ] Both engines start at container boot (check logs: `engine started: XRP`, `engine started: BTC`)
- [ ] Both WebSocket feeds go CONNECTED within 30 s (`tick_manager.get_state(iid) == CONNECTED`)
- [ ] Historical candles load for both instruments (logs: `Hist loaded for XRP: 120 candles`)
- [ ] Candle-close detection fires independently for each instrument (different candle times in logs)
- [ ] LLM signal requests fire for both instruments when auto-trade is ON
- [ ] Toggling auto-trade for BTC does not affect XRP's auto-trade state
- [ ] Signals page shows signals from both instruments, filterable by each
- [ ] P&L tab shows closed trades from both instruments
- [ ] "View" button on Bots page navigates to the correct instrument in Trading tab
- [ ] Stopping one instrument (via Bots page or config `enabled=false`) does not affect the other
- [ ] Container restart restarts both engines (no manual intervention needed)
- [ ] Opening a position on XRP does not block BTC from opening its own position simultaneously

---

## File Change Summary

| File | Action | Key change |
|---|---|---|
| `instruments.toml` | **CREATE** | XRP + BTC config sections |
| `instrument_config.py` | **CREATE** | Config loader, `InstrumentSpec` dataclass |
| `market_data_hub.py` | **REFACTOR** | `dict[iid → HubState]`; shared build thread |
| `trading_engine.py` | **REFACTOR** | `dict[iid → EngineState]`; per-instrument threads |
| `app.py` | **EXTEND** | Boot sequence + new "Bots" page + snapshot reads by iid |
| `tick_manager.py` | no change | Already multi-instrument |
| `trade_manager.py` | no change | Already multi-instrument |
| `signal_worker.py` | no change | Already multi-instrument |
| `signal_log.py` | no change | Already multi-instrument |
| `engine_notify.py` | no change | Already has instrument_id |
| `positions_cache.py` | no change | Returns all positions; filtering already per-iid |
| `Dockerfile` | **1 line** | `COPY instruments.toml .` |

---

## Implementation Order

Implement in this order to maintain a working system at each step:

1. `instruments.toml` + `instrument_config.py` — no runtime impact yet
2. Refactor `market_data_hub.py` — backwards-compatible shims keep Trading tab working
3. Refactor `trading_engine.py` — backwards-compatible shims keep Trading tab working
4. Boot sequence in `app.py` — now both engines start; Trading tab still works
5. "Bots" page in `app.py` — visibility into running engines
6. Remove backwards-compat shims once Bots page is confirmed working
7. P&L tab instrument filter — cosmetic, last
