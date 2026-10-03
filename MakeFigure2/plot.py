# Figure panels
import json
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba,LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter
from model import TIERS,TIER_COLORS,PERIOD_COLORS,fit_curve,write_json

# Figure style
def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
      'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
      'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

# Export figure
def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf',dpi=600,facecolor='white')
    fig.savefig(out/f'{name}.svg',dpi=600,facecolor='white')
    fig.savefig(out/f'{name}.png',dpi=600,facecolor='white')

# Panels a and b: Period comparisons
def boxes(ax,data,metric):
    rng=np.random.default_rng(20240830);bounds=[];audit=[]
    rows=[(tier,cls) for tier in TIERS for cls in ['urban','rural']]
    for j,date in enumerate(['20240830','20241112']):
        for i,(tier,cls) in enumerate(rows):
            x=data.loc[(data.date==date)&data.tier.eq(tier)&data.cls.eq(cls),metric].to_numpy(float)
            x=x[np.isfinite(x)]
            if not len(x):continue
            y=i+(-.19 if j==0 else .19)
            ax.boxplot([x],positions=[y],orientation='horizontal',widths=.31,patch_artist=True,
              whis=1.5,showfliers=False,manage_ticks=False,
              boxprops={'facecolor':to_rgba(PERIOD_COLORS[j],.5),'edgecolor':'black','linewidth':.45},
              medianprops={'color':'black','linewidth':.5},whiskerprops={'color':'black','linewidth':.4},capprops={'color':'black','linewidth':.4})
            q1,q3=np.quantile(x,[.25,.75]);inside=x[(x>=q1-1.5*(q3-q1))&(x<=q3+1.5*(q3-q1))]
            ax.scatter(inside,y+rng.uniform(-.1,.1,len(inside)),s=1.5,color=PERIOD_COLORS[j],alpha=.65,edgecolors='none',zorder=3)
            bounds.extend(inside.tolist());audit.append({'date':date,'tier':tier,'class':cls,'n':len(x),'hidden_points':len(x)-len(inside)})
    ax.set_yticks(range(12),[t+' '+cls.title() for t,cls in rows]);ax.set_ylim(11.65,-.65)
    if bounds:
        low,high=min(bounds),max(bounds);pad=max((high-low)*.05,1e-6);ax.set_xlim(max(0,low-pad),high+pad)
    ax.set_xlabel('Delivery burden per courier (MET h)' if metric=='burden_per_courier' else 'Delivery burden per package (MET h)')
    ax.tick_params(axis='y',length=0,pad=4)
    return audit

# Panels c and d: Tier curves
def curves(ax,data,change=False):
    xcol,ycol=('order_change_pct','burden_change_pct') if change else ('orders','burden')
    x=data[xcol].to_numpy(float);y=data[ycol].to_numpy(float)
    xlim=(-100,200) if change else (0,500);ylim=(-100,200) if change else (5,25)
    finite=np.isfinite(x)&np.isfinite(y)
    visible=finite&(x>=xlim[0])&(x<=xlim[1])&(y>=ylim[0])&(y<=ylim[1])
    ax.scatter(x[visible],y[visible],s=.45,color='#85AEC9',alpha=.15,edgecolors='none',rasterized=True,zorder=1)
    stats=[];output=[]
    for tier,color in zip(TIERS,TIER_COLORS):
        mask=finite&data.tier.eq(tier).to_numpy()
        if mask.sum()<8 or len(np.unique(x[mask]))<4:
            stats.append({'tier':tier,'n':int(mask.sum()),'fit':False});continue
        xx,yy,audit=fit_curve(x[mask],y[mask],xlim,change)
        ax.plot(xx,yy,color=color,lw=1,label=tier,zorder=4)
        audit['tier']=tier;stats.append(audit)
        output.extend({'tier':tier,'x':float(a),'y':float(b)} for a,b in zip(xx,yy))
    ax.set(xlim=xlim,ylim=ylim)
    if change:
        ax.axhline(0,color='#777777',lw=.4,ls='--',zorder=2)
        ax.axvline(0,color='#AAAAAA',lw=.35,ls=':',zorder=2)
        ax.plot(xlim,xlim,color='#444444',lw=.5,ls='--',zorder=3)
        ax.text(.78,.91,'1:1',transform=ax.transAxes)
        ax.set_xlabel('Order volume change (%)');ax.set_ylabel('Delivery burden change (%)')
        ax.set_box_aspect(1);ax.legend(loc='center left',bbox_to_anchor=(1.06,.5),fontsize=7,handlelength=1.6)
    else:
        ax.set_xlabel('Order volume');ax.set_ylabel('Delivery burden per courier (MET h)')
        ax.set_xticks(np.arange(0,501,100));ax.set_yticks(np.arange(5,26,5))
    return {'input_rows':len(data),'nonfinite':int((~finite).sum()),'visible':int(visible.sum()),
      'missing_tier':int(data.tier.isna().sum()),'fits':stats},output

# Map polygons
def polygons(geometry):
    if geometry.geom_type=='Polygon':yield geometry
    elif hasattr(geometry,'geoms'):
        for part in geometry.geoms:yield from polygons(part)

# Panels e and f: City maps
def city_map(fig,ax,data,features,cmap,norm):
    values=data.set_index('city').burden_per_order.to_dict();verts=[];colors=[]
    for feature in features:
        city=feature['properties']['市'];v=values.get(city,np.nan)
        color=cmap(norm(v)) if np.isfinite(v) and v>0 else '#E6E6E6'
        for p in polygons(shapely.make_valid(shape(feature['geometry']))):
            verts.append(np.asarray(p.exterior.coords));colors.append(color)
    # Draw map plane
    def draw(a,limits,inset=False):
        a.add_collection(PolyCollection(verts,facecolors=colors,edgecolors='#FFFFFF',linewidths=.13))
        a.set(xlim=limits[:2],ylim=limits[2:]);a.set_aspect(1/np.cos(np.deg2rad(35)))
        a.set_xticks([]);a.set_yticks([])
        for sp in a.spines.values():sp.set_visible(inset);sp.set_linewidth(.45)
    draw(ax,(73,136,17,54))
    sea=ax.inset_axes([.84,.025,.14,.32]);draw(sea,(105,125,2,25),True)
    cax=ax.inset_axes([.18,-.065,.60,.017])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-2,-2));formatter.set_useOffset(False)
    cb.formatter=formatter;cb.update_ticks();cb.ax.tick_params(labelsize=7,length=2,pad=2)
    cb.set_label('Delivery burden per order (MET h)',fontsize=8,labelpad=3)
    cb.ax.yaxis.set_offset_position('right')
    cax.text(-.015,.5,'Low',ha='right',va='center',transform=cax.transAxes,fontsize=7)
    cax.text(1.015,.5,'High',ha='left',va='center',transform=cax.transAxes,fontsize=7)

# Compose figure
def make_figure(work,boundaries,out):
    style()
    cities=pd.concat([pd.read_csv(work/f'city_classes_{date}.csv',dtype={'date':str}) for date in ['20240830','20241112']],ignore_index=True)
    couriers=pd.read_csv(work/'couriers_20240830.csv',dtype={'userid':str})
    changes=pd.read_csv(work/'courier_changes.csv',dtype={'userid':str})
    maps=[pd.read_csv(work/f'city_map_{date}.csv') for date in ['20240830','20241112']]
    features=json.loads(boundaries.read_text(encoding='utf-8'))['features']
    v=np.concatenate([d.burden_per_order.to_numpy(float) for d in maps]);v=v[np.isfinite(v)&(v>0)]
    if not len(v):raise ValueError('No city map values')
    norm=Normalize(float(v.min()),float(v.max()) if v.max()>v.min() else float(v.min()+1e-6))
    cmap=LinearSegmentedColormap.from_list('burden',['#EAE057','#EBB353','#DC855A','#C5616D','#B44B7C','#943C83','#492F83'])
    fig=plt.figure(figsize=(9,9.3))
    positions={'a':[.16,.66,.325,.30],'b':[.57,.66,.325,.30],
      'c':[.10,.425,.40,.19],'d':[.60,.425,.205,.19],
      'e':[.035,.068,.445,.305],'f':[.525,.068,.445,.305]}
    axes={p:fig.add_axes(pos) for p,pos in positions.items()};audit={}
    audit['a']=boxes(axes['a'],cities,'burden_per_courier')
    audit['b']=boxes(axes['b'],cities,'burden_per_order')
    handles=[Patch(facecolor=to_rgba(col,.5),edgecolor='black',linewidth=.4,label=label) for col,label in zip(PERIOD_COLORS,['Regular periods','Promotion periods'])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,fontsize=8)
    audit['c'],c=curves(axes['c'],couriers)
    audit['d'],d=curves(axes['d'],changes,True)
    pd.DataFrame(c).to_csv(out/'panel_c_curves.csv',index=False)
    pd.DataFrame(d).to_csv(out/'panel_d_curves.csv',index=False)
    for p,frame in zip(['e','f'],maps):city_map(fig,axes[p],frame,features,cmap,norm)
    for p,pos in positions.items():fig.text(.04 if p in 'ace' else .485,pos[1]+pos[3]-.009,p,fontweight='bold',fontsize=12)
    save(fig,out,'Figure2');plt.close(fig)
    write_json(out/'plot_audit.json',audit)
