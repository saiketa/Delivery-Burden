# Figure panels
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,ScalarFormatter,NullLocator
from metrics import spearman,lowess,aoi_density

# Plot colors
TIERS=['Tier 1','New Tier 1','Tier 2','Tier 3','Tier 4','Tier 5']
TIER_COLORS=['#27644D','#6F9C89','#B6CEC4','#C3B5D3','#9576B5','#614098']
AREA_COLORS={'urban':'#7DB3D0','rural':'#F4AD83'}
BOX_COLORS={'urban':'#356680','rural':'#985A35'}
INK='#55436D'

# Figure style
def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
      'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
      'axes.linewidth':.65,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})

# Panels a, c, e: Raincloud distributions
def raincloud(ax,data,metric,ylabel,legend):
    rng=np.random.default_rng(20240830);values=[];audit=[]
    for i,tier in enumerate(TIERS,1):
        for cls,offset in [('urban',-.19),('rural',.19)]:
            raw=data.loc[data.tier.eq(tier)&data.cls.eq(cls),metric].to_numpy(float)
            v=raw[np.isfinite(raw)];pos=i+offset
            if not len(v):continue
            values.extend(v.tolist());base=AREA_COLORS[cls];dark=BOX_COLORS[cls]
            if len(v)>1 and np.ptp(v)>0:
                violin=ax.violinplot([v],positions=[pos-.035],widths=.32,showmeans=False,
                  showmedians=False,showextrema=False,bw_method='scott',points=150)
                for body in violin['bodies']:
                    body.set_facecolor(base);body.set_edgecolor('none');body.set_alpha(.5)
                    vertices=body.get_paths()[0].vertices;vertices[:,0]=np.minimum(vertices[:,0],pos-.035)
            ax.scatter(pos+rng.uniform(.075,.18,len(v)),v,s=3.0,color=base,alpha=.65,edgecolors='none',zorder=2)
            ax.boxplot([v],positions=[pos+.015],widths=.055,patch_artist=True,showfliers=False,
              whis=1.5,manage_ticks=False,boxprops={'facecolor':dark,'edgecolor':dark,'linewidth':.5},
              medianprops={'color':dark,'linewidth':.5},whiskerprops={'color':dark,'linewidth':.5},
              capprops={'color':dark,'linewidth':.5},zorder=3)
            ax.scatter([pos+.015],[np.median(v)],s=3,color='white',edgecolors='none',zorder=4)
            audit.append({'tier':tier,'class':cls,'n':len(v),'nonfinite':len(raw)-len(v)})
    if values:
        lo,hi=min(values),max(values);pad=max((hi-lo)*.06,1e-6);ax.set_ylim(max(0,lo-pad),hi+pad)
    ax.set_xticks(range(1,7),TIERS,rotation=25,ha='right');ax.set_xlabel('City tier');ax.set_ylabel(ylabel)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#DDDDDD',linestyle=':',linewidth=.5)
    ax.legend(handles=[Patch(facecolor=AREA_COLORS[c],label=c.title()) for c in ['urban','rural']],loc=legend)
    return audit

# Correlation label
def correlation_label(ax,rho):
    label=f'Spearman ρ = {rho:.2f}' if rho is not None else 'Spearman ρ = undefined'
    ax.text(.97,.06,label,ha='right',va='bottom',transform=ax.transAxes,
      bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':2})

# Panels b and d: City correlations
def city_correlation(ax,data,xcol,ycol,xlabel,ylabel):
    x=data[xcol].to_numpy(float);y=data[ycol].to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0);x=x[ok];y=y[ok]
    if len(x)<3:raise ValueError('Insufficient positive city pairs')
    tier=data.loc[ok,'tier'].to_numpy();rho=spearman(x,y)
    for label,color in zip(TIERS,TIER_COLORS):
        mask=tier==label
        ax.scatter(x[mask],y[mask],s=7,facecolors=color,edgecolors=INK,linewidths=.15,alpha=.4)
    unknown=~pd.Series(tier).isin(TIERS).to_numpy()
    ax.scatter(x[unknown],y[unknown],s=7,color='#AAAAAA',alpha=.4,edgecolors='none')
    grid=np.linspace(np.log10(x.min()),np.log10(x.max()),300)
    fitted=lowess(np.log10(x),y,grid);ax.plot(10**grid,fitted,color=INK,lw=1.1)
    ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5))
    formatter=ScalarFormatter();formatter.set_scientific(False);formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);correlation_label(ax,rho)
    return {'n':len(x),'excluded_nonpositive_nonfinite':int((~ok).sum()),'rho':rho,'unknown_tier':int(unknown.sum())}

# Panel f: AOI density correlation
def aoi_panel(fig,ax,data,out):
    x=data.population_persons.to_numpy(float);y=data.burden.to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0);x=x[ok];y=y[ok]
    if len(x)<3:raise ValueError('Insufficient positive AOI pairs')
    points,density,grid,fitted,ids=aoi_density(x,y)
    cmap=LinearSegmentedColormap.from_list('blue_density',['#F7FBFF','#C6DBEF','#6BAED6','#2171B5','#08306B'])
    norm=Normalize(0,float(density.max()))
    marks=ax.scatter(points[:,0],points[:,1],c=density,s=1.7,cmap=cmap,norm=norm,edgecolors='none',rasterized=True)
    ax.plot(grid,fitted,color='#87BED1',lw=1.1)
    ax.set_xlim(np.log10(1+x.min()),np.log10(1+x.max()))
    lo,hi=np.log10(y.min()),np.log10(y.max());ax.set_ylim(lo,hi)
    powers=np.arange(np.floor(np.log10(x.min())),np.ceil(np.log10(x.max()))+1)
    ticks=10.**powers;ticks=ticks[(ticks>=x.min())&(ticks<=x.max())]
    ax.set_xticks(np.log10(1+ticks),[f'{v:,.0f}' if v>=1 else f'{v:g}' for v in ticks])
    ax.set_xlabel('AOI population (persons)');ax.set_ylabel('AOI delivery burden\n(log$_{10}$ MET h)')
    rho=spearman(x,y);correlation_label(ax,rho)
    cax=ax.inset_axes([.32,-.28,.42,.035]);cb=fig.colorbar(marks,cax=cax,orientation='horizontal')
    cb.set_ticks([0,density.max()],labels=['Low','High']);cb.set_label('Probability density',labelpad=2)
    source_rows=np.flatnonzero(ok)[ids]
    pd.DataFrame({'source_row':source_rows,'population_persons':x[ids],'burden':y[ids],'density':density}).to_csv(out/'displayed_aoi_points.csv',index=False)
    return {'n':len(x),'displayed':len(points),'excluded_nonpositive_nonfinite':int((~ok).sum()),'rho':rho,'seed':20240830}

# Compose figure
def make_figure(work,out):
    style();cities=pd.read_csv(work/'raincloud.csv')
    urban=pd.read_csv(work/'urban_correlation.csv');city=pd.read_csv(work/'city_correlation.csv')
    aoi=pd.read_csv(work/'aoi_population_burden.csv',usecols=['population_persons','burden'])
    fig=plt.figure(figsize=(12,12))
    positions={'a':[.09,.715,.43,.225],'b':[.66,.715,.30,.225],
      'c':[.09,.405,.43,.225],'d':[.66,.405,.30,.225],
      'e':[.09,.095,.43,.225],'f':[.66,.095,.30,.225]}
    axes={p:fig.add_axes(pos) for p,pos in positions.items()};audit={}
    for p,metric,ylabel,legend in [
      ('a','burden_per_courier','Delivery burden per courier\n(MET h)','lower left'),
      ('c','burden_per_order','Delivery burden per package\n(MET h)','upper left'),
      ('e','burden_per_aoi','Mean delivery burden per AOI\n(MET h)','upper right')]:
        audit[p]=raincloud(axes[p],cities,metric,ylabel,legend)
    audit['b']=city_correlation(axes['b'],urban,'urban_population_10k','burden_per_courier',
      'Urban population (10$^4$ persons)','Urban delivery burden\nper courier (MET h)')
    audit['d']=city_correlation(axes['d'],city,'population_10k','burden_per_order',
      'City population (10$^4$ persons)','City delivery burden\nper package (MET h)')
    audit['f']=aoi_panel(fig,axes['f'],aoi,out)
    for p,pos in positions.items():fig.text(pos[0]-.065,pos[1]+pos[3]+.012,p,fontweight='bold',fontsize=14)
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=3.5,markerfacecolor=c,markeredgecolor=INK,markeredgewidth=.2,label=t) for t,c in zip(TIERS,TIER_COLORS)]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.08,.006),ncol=6,columnspacing=1.0,handlelength=.6,handletextpad=.4)
    fig.savefig(out/'Figure3.pdf',dpi=600,facecolor='white')
    fig.savefig(out/'Figure3.svg',dpi=600,facecolor='white')
    fig.savefig(out/'Figure3.png',dpi=600,facecolor='white')
    plt.close(fig)
    (out/'plot_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
