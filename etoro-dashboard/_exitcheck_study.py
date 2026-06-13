"""exit_check_tf study: does a faster CLOSED-LTF signal-reversal exit help trend
plans? Baseline (exit-check = interval) vs ½-interval, same entries + same exit
params, on the overlapping window (HTF clipped to LTF coverage). Backtest-only."""
import os
import pandas as pd
import backtester as bt
from etoro_client import get_shared_client

IIDS = {'XRP  (XRP)': 100003, 'Bitcoin  (BTC)': 100000,
        'Tesla Motors, Inc.  (TSLA)': 1111, 'NVIDIA Corporation  (NVDA)': 1137,
        'Gold (Non Expiry)  (GOLD)': 18, 'Oil (Non Expiry)  (OIL)': 17}
HALF = {3600: 1800, 1800: 900}                       # interval -> half-interval LTF
TREND = ['supertrend', 'macd', 'ma_crossover', 'donchian', 'ichimoku', 'daviddtech', 'adx']
WIN = 120
c = get_shared_client(os.environ['ETORO_API_KEY'], os.environ['ETORO_USER_KEY'])

def oos(res):
    _, o = res.oos_split()
    return o['pf'], o['n'], res.summary()['pnl'], res.summary()['n']

print(f"{'strat':12} {'asset':6} {'ivl':4} | {'base pnl':>8} {'oosPF':>5} {'n':>3} | {'½TF pnl':>8} {'oosPF':>5} {'n':>3} | verdict")
cache = {}
for label, iid in IIDS.items():
    for secs in (3600, 1800):
        ltf = HALF[secs]
        for strat in TREND:
            try:
                if (iid, secs) not in cache:
                    cache[(iid, secs)] = c.get_hist_candles(iid, secs, 1000)
                if (iid, ltf) not in cache:
                    cache[(iid, ltf)] = c.get_hist_candles(iid, ltf, 1000)
                htf, lt = cache[(iid, secs)], cache[(iid, ltf)]
                if htf is None or lt is None or len(lt) < 200: continue
                htf = htf.copy(); lt = lt.copy()
                htf['time'] = pd.to_datetime(htf['time'], utc=True)
                lt['time'] = pd.to_datetime(lt['time'], utc=True)
                # clip HTF to LTF coverage so every tested HTF bar has LTF data
                htf = htf[htf['time'] >= lt['time'].iloc[0]].reset_index(drop=True)
                if len(htf) < WIN + 30: continue
                hsig = bt.compute_signal_series(htf, strat, iid, 0.05, window_bars=WIN)
                lsig = bt.compute_signal_series(lt, strat, iid, 0.05, window_bars=WIN)
                if hsig is None or lsig is None: continue
                params = dict(stop_mult=2.5, trail_mult=3.0, tp_pct=0.0, window_bars=WIN)
                base = bt.simulate_exits(htf, hsig, strat, label, secs, **params)
                if base.summary()['n'] < 8: continue
                revmap = bt.build_exit_rev_by_bar(htf, lt, lsig)
                mtf = bt.simulate_exits(htf, hsig, strat, label, secs, exit_rev_by_bar=revmap, **params)
                bpf, bn, bpnl, bN = oos(base)
                mpf, mn, mpnl, mN = oos(mtf)
                better = "FASTER WINS" if (mpnl > bpnl + 1 and mpf >= bpf) else ("~same" if abs(mpnl-bpnl) <= 1 else "slower better")
                bpf_s = 99 if bpf==float('inf') else bpf; mpf_s = 99 if mpf==float('inf') else mpf
                print(f"{strat:12} {label.split()[0][:6]:6} {secs//60:>3}m | {bpnl:>8.2f} {bpf_s:>5.1f} {bN:>3} | {mpnl:>8.2f} {mpf_s:>5.1f} {mN:>3} | {better}", flush=True)
            except Exception as e:
                pass
print("STUDY DONE", flush=True)
