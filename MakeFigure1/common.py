# Shared calculations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

# Activity parameters
ACTIVITIES = ['still', 'walking', 'driving', 'upstairs', 'downstairs']
MET = np.array([1.3, 3.0, 2.0, 6.8, 3.5])
COLORS = ['#303030', '#49AD78', '#398BC0', '#E85A65']

# Path
def path(c, key):
    p = Path(c[key]).expanduser()
    return p if p.is_absolute() else c['_base'] / p

# Write json
def write_json(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

# Normalize
def normalize(s):
    return s.astype('string').fillna('').str.strip().str.replace(r'\.0$', '', regex=True)

# Burden
def burden(seconds):
    x = np.asarray(seconds, float)
    if x.shape[-1] != 5 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected five finite nonnegative durations: still, walking, driving, upstairs, downstairs')
    return x @ MET / 3600

# Assign courier groups
def group_id(uid, date='2024-08-30', groups=800):
    prefix = 'minute-band-' + date.replace('-', '') + ':'
    return int.from_bytes(hashlib.blake2b((prefix+str(uid)).encode(), digest_size=8).digest(), 'big') % groups

# Weighted quantiles
def weighted_quantile(x, w, q):

    x, w = np.asarray(x), np.asarray(w)
    if not np.isfinite(w).all() or (w <= 0).any() or (w != np.round(w)).any():
        raise ValueError('Positive integer frequency weights required')
    i = np.argsort(x); x = x[i]; cw = np.cumsum(w[i].astype('int64'))
    r = (int(cw[-1])-1)*q; lo = int(np.floor(r)); hi = int(np.ceil(r))
    return float((1-(r-lo))*x[np.searchsorted(cw, lo, side='right')] + (r-lo)*x[np.searchsorted(cw, hi, side='right')])

# Histogram
def histogram(x, w=None, bins=100):
    x = np.asarray(x, float)
    w = np.ones(len(x), dtype='int64') if w is None else np.asarray(w)
    keep = np.isfinite(x) & (x > 0) & np.isfinite(w) & (w > 0)
    invalid_rows = int((~keep).sum()); x, w = x[keep], w[keep]
    if not len(x): raise ValueError('No positive finite burdens')
    q1, q3 = [weighted_quantile(x, w, q) for q in (.25, .75)]
    lo, hi = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    keep = (x >= lo) & (x <= hi); v, weights = x[keep], w[keep]
    a, b = v.min(), v.max()
    if a == b: a, b = max(0., a-.5), b+.5
    edges = np.linspace(a, b, bins+1)
    counts = np.histogram(v, bins=edges, weights=weights)[0]
    assert int(counts.sum()) == int(weights.sum())
    return pd.DataFrame({'bin_left_MET_h':edges[:-1], 'bin_right_MET_h':edges[1:], 'count':counts}), dict(
        before=int(w.sum()), retained=int(weights.sum()), excluded_IQR=int(w[~keep].sum()), invalid_rows=invalid_rows,
        q1=q1, q3=q3, lower=lo, upper=hi, bins=bins)

# Minute summaries
def minute_sources(counts, sums, up, orders, start=420):

    if counts.shape != sums.shape[:2] or sums.shape[2] != 4 or up.shape != counts.shape:
        raise ValueError('Minute array shapes do not match')
    assert np.isfinite(sums).all() and (sums >= 0).all() and (up >= 0).all()
    down = sums[:,:,3]-up
    assert down.min() >= -1e-7
    five = np.stack([sums[:,:,0], sums[:,:,1], sums[:,:,2], up, np.maximum(down,0)], axis=-1)
    group_burden = burden(five)
    n = counts.sum(axis=1)
    if (n <= 0).any(): raise ValueError('Minute without observed couriers')
    groupmeans = np.divide(sums, counts[:,:,None], out=np.full_like(sums,np.nan), where=counts[:,:,None]>0)
    mean = sums.sum(axis=1)/n[:,None]
    t = np.arange(start, start+len(n))
    frame = pd.DataFrame({'minute':[f'{m//60:02}:{m%60:02}' for m in t], 'observed_couriers':n,
        **{name+'_mean':mean[:,j] for j,name in enumerate(['Stationary','Walking','Driving','Stairs'])},
        'burden_mean_MET_h':group_burden.sum(axis=1)/n, 'order_events':orders[t]})
    if counts.shape[1] != 800: raise ValueError('Figure contract requires 800 activity groups')
    c100 = counts.reshape(len(n),8,100).sum(axis=1)
    b100 = group_burden.reshape(len(n),8,100).sum(axis=1)
    bmeans = np.divide(b100,c100,out=np.full_like(b100,np.nan),where=c100>0)
    return frame, groupmeans, bmeans
