# Shared calculations
import json
from pathlib import Path
import numpy as np

# Activity parameters
ACTIVITIES=['still','walking','driving','upstairs','downstairs']
MET=np.array([1.3,3.0,2.0,6.8,3.5])
TIERS=['Tier 1','New Tier 1','Tier 2','Tier 3','Tier 4','Tier 5']
TIER_COLORS=['#DFA62B','#9467BD','#129C83','#27B4BC','#187EAD','#E56B57']
PERIOD_COLORS=['#7DB3D0','#F4AD83']

# Normalize identifiers
def normalize(s):
    return s.astype('string').fillna('').str.strip().str.replace(r'\.0$','',regex=True)

# City names
def city_name(source):
    return Path(source).stem.removesuffix('_AOI_relation').removesuffix('_AOI')

# Tier labels
def tier_label(value):
    value=str(value).strip()
    mapping=dict(zip(['一线城市','新一线城市','二线城市','三线城市','四线城市','五线城市'],TIERS))
    result=mapping.get(value,value)
    if result not in TIERS and result not in ['', 'nan','<NA>']:
        raise ValueError('Invalid tier: '+value)
    return result if result in TIERS else None

# Delivery burden
def burden(seconds):
    x=np.asarray(seconds,float)
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid activity durations')
    return x@MET/3600

# Write audit
def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

# Penalized tier curves
def fit_curve(x,y,view,change=False):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<8 or len(np.unique(x))<4:raise ValueError('Insufficient distinct observations')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite fit input')
    if not change and (x<0).any():raise ValueError('Negative order volume')
    transform=(lambda v:np.arcsinh(v/100)) if change else np.log1p
    raw=transform(x);lo=raw.min() if change else 0.;span=raw.max()-lo
    if span<=0:raise ValueError('Constant predictor')
    z=(raw-lo)/span;knots=np.unique(np.quantile(z,np.linspace(.05,.95,10)))
    # Spline basis
    def design(v):
        return np.column_stack([np.ones_like(v),v,v*v,v*v*v]+[np.maximum(v-k,0)**3 for k in knots])
    X=design(z);A=X.T@X;b=X.T@y;grid=np.linspace(0,1,600)
    D2=np.column_stack([np.zeros_like(grid),np.zeros_like(grid),np.full_like(grid,2),6*grid]+[6*np.maximum(grid-k,0) for k in knots])
    penalty=D2.T@D2/600;best=None
    for lam in np.logspace(-7,5,65):
        M=A+lam*penalty+np.eye(A.shape[0])*1e-10
        coef=np.linalg.solve(M,b);edf=np.trace(np.linalg.solve(M,A));res=y-X@coef;sse=float(res@res)
        if edf>=len(y):continue
        gcv=len(y)*sse/(len(y)-edf)**2
        if best is None or gcv<best[0]:best=(gcv,coef,lam,edf,sse)
    if best is None:raise ValueError('No valid smoothing parameter')
    _,coef,lam,edf,sse=best
    left=max(view[0],x.min());right=min(view[1],x.max())
    if left>=right:raise ValueError('No fitted range in viewport')
    xx=np.linspace(left,right,601) if change else np.arange(np.ceil(left),np.floor(right)+1)
    yy=design((transform(xx)-lo)/span)@coef
    variance=float(np.sum((y-y.mean())**2))
    return xx,yy,{'n':len(x),'lambda':float(lam),'effective_df':float(edf),'R2':1-sse/variance if variance>0 else None}
