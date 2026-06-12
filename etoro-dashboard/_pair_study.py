"""Phase 0 — XRP/BTC pairs-trading feasibility study (evidence gate).

Rolling-beta log-spread z-score strategy, dollar-neutral legs, BOTH legs'
spreads modeled, conservative next-bar-open fills, 70/30 trade-split OOS.
"""
import os
import numpy as np
import pandas as pd
from etoro_client import get_shared_client

XRP, BTC = 100003, 100000
SPREAD = {XRP: 0.08, BTC: 0.05}          # % round-trip cost is 2x half each leg
LEG_USD = 1000.0

c = get_shared_client(os.environ['ETORO_API_KEY'], os.environ['ETORO_USER_KEY'])

def aligned(secs):
    a = c.get_hist_candles(XRP, secs, 1000)
    b = c.get_hist_candles(BTC, secs, 1000)
    if a is None or b is None: return None
    a = a[['time','Open','Close']].rename(columns={'Open':'xo','Close':'xc'})
    b = b[['time','Open','Close']].rename(columns={'Open':'bo','Close':'bc'})
    m = a.merge(b, on='time', how='inner').reset_index(drop=True)
    return m if len(m) > 400 else None

def simulate(m, w_beta, w_z, z_in, z_out, z_stop, timeout):
    lx, lb = np.log(m['xc'].values), np.log(m['bc'].values)
    n = len(m)
    # rolling beta (cov/var) + rolling z of spread — all trailing, no lookahead
    beta = np.full(n, np.nan)
    for i in range(w_beta, n):
        xs, bs = lx[i-w_beta:i], lb[i-w_beta:i]
        vb = bs.var()
        beta[i] = (np.cov(xs, bs)[0,1] / vb) if vb > 0 else np.nan
    S = lx - np.where(np.isnan(beta), 0, beta) * lb
    z = np.full(n, np.nan)
    for i in range(w_beta + w_z, n):
        win = S[i-w_z:i]
        sd = win.std()
        z[i] = (S[i] - win.mean()) / sd if sd > 0 else 0.0
    xo, bo = m['xo'].values, m['bo'].values
    trades, pos = [], 0          # +1 long-spread (long XRP short BTC), -1 short-spread
    ex_px = en_px = None; en_i = 0
    hx, hb = SPREAD[XRP]/200.0, SPREAD[BTC]/200.0   # half-spread fractions
    pend = 0
    for i in range(w_beta + w_z, n - 1):
        if pend and pos == 0:
            d = pend; pend = 0
            # fills at NEXT bar open with per-leg half-spreads
            ex_px = (xo[i] * (1 + hx) if d > 0 else xo[i] * (1 - hx))
            en_px = (bo[i] * (1 - hb) if d > 0 else bo[i] * (1 + hb))
            pos, en_i = d, i
        if pos != 0:
            zi = z[i]
            done = abs(zi) <= z_out or abs(zi) >= z_stop or (i - en_i) >= timeout
            if done or i == n - 2:
                xq = xo[i+1] * (1 - hx) if pos > 0 else xo[i+1] * (1 + hx)
                bq = bo[i+1] * (1 + hb) if pos > 0 else bo[i+1] * (1 - hb)
                px = LEG_USD * (xq / ex_px - 1) * (1 if pos > 0 else -1)
                pb = LEG_USD * (bq / en_px - 1) * (-1 if pos > 0 else 1)
                reason = ('break' if abs(zi) >= z_stop else
                          'timeout' if (i - en_i) >= timeout else 'converged')
                trades.append((px + pb, reason, i - en_i))
                pos = 0
        elif not pend:
            zi = z[i]
            if not np.isnan(zi):
                if zi <= -z_in: pend = +1
                elif zi >= z_in: pend = -1
    return trades

def pf(pnls):
    g = sum(p for p in pnls if p > 0); l = abs(sum(p for p in pnls if p <= 0))
    return 99.0 if l == 0 and g > 0 else (g / l if l else 0.0)

def half_life(m, w_beta=200):
    lx, lb = np.log(m['xc'].values), np.log(m['bc'].values)
    xs, bs = lx[-600:], lb[-600:]
    b = np.cov(xs, bs)[0,1] / bs.var()
    S = xs - b * bs
    dS, S1 = np.diff(S), S[:-1]
    phi = np.cov(dS, S1)[0,1] / S1.var()
    return -np.log(2) / phi if phi < 0 else float('inf')

print(f"{'ivl':4} {'wz':>4} {'zin':>4} {'zout':>4} {'tmo':>4} {'n':>4} {'win%':>5} {'pnl$':>8} {'isPF':>5} {'oosPF':>6} {'oosN':>4} {'avg hold':>8}")
for secs, ivl in ((900,'15m'), (1800,'30m'), (3600,'1h')):
    m = aligned(secs)
    if m is None: print(f'{ivl}: no aligned data'); continue
    hl = half_life(m)
    print(f'-- {ivl}: {len(m)} aligned bars · spread half-life ≈ {hl:.0f} bars --')
    for w_z in (60, 100, 150):
        for z_in in (1.5, 2.0, 2.5):
            for z_out in (0.25, 0.5):
                for tmo in (150, 300):
                    tr = simulate(m, 200, w_z, z_in, z_out, 4.0, tmo)
                    if len(tr) < 8: continue
                    pnls = [t[0] for t in tr]
                    cut = max(1, int(len(pnls) * 0.7))
                    ins, oos = pnls[:cut], pnls[cut:]
                    if len(oos) < 3: continue
                    print(f"{ivl:4} {w_z:>4} {z_in:>4} {z_out:>4} {tmo:>4} {len(tr):>4} "
                          f"{100*sum(1 for p in pnls if p>0)/len(pnls):>5.1f} {sum(pnls):>8.2f} "
                          f"{pf(ins):>5.2f} {pf(oos):>6.2f} {len(oos):>4} "
                          f"{np.mean([t[2] for t in tr]):>8.1f}", flush=True)
print('PAIR STUDY DONE')
