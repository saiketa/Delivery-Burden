# Statistical functions
import numpy as np
import pandas as pd

# Spearman correlation
def spearman(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))

# Local linear smoothing
def lowess(x,y,grid,fraction=.55):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid smoothing input')
    count=min(len(x),max(3,int(np.ceil(fraction*len(x)))))
    fitted=np.empty(len(grid))
    for i,point in enumerate(grid):
        distance=np.abs(x-point);width=float(np.partition(distance,count-1)[count-1])
        if width<=0:width=float(distance.max()) or 1.
        w=(1-np.minimum(distance/width,1)**3)**3;dx=x-point
        s0=w.sum();s1=np.sum(w*dx);s2=np.sum(w*dx*dx);t0=np.sum(w*y);t1=np.sum(w*dx*y)
        determinant=s0*s2-s1*s1
        if s0<=0:raise ValueError('Zero smoothing weights')
        fitted[i]=(t0*s2-t1*s1)/determinant if determinant>1e-14*max(s0*s2,1) else t0/s0
    return fitted

# AOI density
def aoi_density(population,burden):
    xy=np.c_[np.log10(1+population),np.log10(burden)]
    if np.ptp(xy[:,0])==0 or np.ptp(xy[:,1])==0:raise ValueError('Constant density coordinate')
    hist,xe,ye=np.histogram2d(xy[:,0],xy[:,1],bins=180)
    kernel=np.exp(-.5*(np.arange(-6,7)/2)**2);kernel/=kernel.sum()
    smooth=np.apply_along_axis(lambda a:np.convolve(a,kernel,mode='same'),0,hist)
    smooth=np.apply_along_axis(lambda a:np.convolve(a,kernel,mode='same'),1,smooth)
    smooth/=smooth.sum()*(xe[1]-xe[0])*(ye[1]-ye[0])
    ids=np.random.default_rng(20240830).choice(len(xy),min(20000,len(xy)),replace=False)
    points=xy[ids];ix=np.clip(np.searchsorted(xe,points[:,0],side='right')-1,0,179)
    iy=np.clip(np.searchsorted(ye,points[:,1],side='right')-1,0,179)
    density=smooth[ix,iy];order=np.argsort(density)
    grid=np.linspace(xy[:,0].min(),xy[:,0].max(),300)
    return points[order],density[order],grid,lowess(xy[:,0],xy[:,1],grid),ids[order]
