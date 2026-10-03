# Figure panels
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from common import COLORS, path, minute_sources, write_json
from spatial import plot_a, plot_b, prepare_b

NAMES=['Stationary','Walking','Driving','Stairs']

# Style
def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,'legend.frameon':False})

# Save
def save(fig,out,name,dpi=400):
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=dpi,facecolor='white')

# Compress
def compress(y):
    y=np.asarray(y);return np.where(y<=18,y,np.where(y<27,18+(y-18)/9,y-8))

# Expand
def expand(y):
    y=np.asarray(y);return np.where(y<=18,y,np.where(y<19,18+(y-18)*9,y+8))

# Background
def background(ax,orders,start,end):
    cmap=LinearSegmentedColormap.from_list('order_volume',[(124/255,183/255,229/255,0),(124/255,183/255,229/255,1)])
    norm=Normalize(0,max(1,float(np.max(orders))))
    ax.imshow(np.asarray(orders)[None,:],cmap=cmap,norm=norm,extent=(start/60,end/60,0,1),
              transform=ax.get_xaxis_transform(),aspect='auto',interpolation='nearest',zorder=0,rasterized=True)
    return ScalarMappable(norm=norm,cmap=cmap)

# Time axes
def time_axes(ax,start,end):
    ticks=np.arange(np.ceil(start/60),end/60+1)
    ax.set_xlim(start/60,end/60);ax.set_xticks(ticks);ax.set_xticklabels([f'{int(h):02}:00' for h in ticks])
    for sp in ax.spines.values():sp.set_visible(True)

# Panel c: Activity durations
def plot_c(ax,data,groups,start,end):
    x=(np.arange(start,end)+.5)/60
    m=background(ax,data.order_events,start,end)
    for j,(name,color) in enumerate(zip(NAMES,COLORS)):
        light=tuple(.6+.4*v for v in to_rgb(color))
        ax.scatter(np.repeat(x,800),groups[:,:,j].ravel(),s=.55,color=light,alpha=.045,edgecolors='none',rasterized=True,zorder=2)
        ax.plot(x,data[name+'_mean'],color=color,lw=.85,label=name,zorder=3)
    ax.set_yscale('function',functions=(compress,expand));ax.set_ylim(0,55);ax.set_yticks([0,10,30,40,50])
    ax.set_ylabel('Activity duration\nper courier (s)');time_axes(ax,start,end)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=4,columnspacing=2.5,fontsize=10)
    y=float(compress(22.5)/compress(55))
    for x in [0,1]:
        ax.plot([x,x],[y-.024,y+.024],transform=ax.transAxes,color='white',lw=3,clip_on=False,zorder=8)
        for d in [-.008,.008]:ax.plot([x-.004,x+.004],[y+d-.006,y+d+.006],transform=ax.transAxes,color='black',lw=.6,clip_on=False,zorder=9)
    return m

# Panel d: Minute burden
def plot_d(ax,data,groups,start,end):
    x=(np.arange(start,end)+.5)/60
    m=background(ax,data.order_events,start,end)
    ax.scatter(np.repeat(x,100),groups.ravel(),s=1.0,color='#777777',alpha=.18,edgecolors='none',rasterized=True,zorder=2)
    ax.plot(x,data.burden_mean_MET_h,color='#303030',lw=.85,zorder=3)
    ax.set_ylim(.023992688302870546,.0280);ax.set_yticks([.0245,.0255,.0265,.0275])
    fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((-2,-2));fmt.set_useOffset(False);ax.yaxis.set_major_formatter(fmt)
    ax.set_ylabel('Delivery burden\nper courier (MET h)');ax.set_xlabel('Time of day');time_axes(ax,start,end)
    return m

# Panels e and f: Burden distributions
def plot_hist(ax,d,panel):
    counts=d['count'].to_numpy();edges=np.r_[d.bin_left_MET_h,d.bin_right_MET_h.iloc[-1]]
    base=np.array(to_rgb('#9870CC' if panel=='e' else '#DCA185'))
    colors=1-(.2+.8*counts/max(1,counts.max()))[:,None]*(1-base)
    ax.bar(edges[:-1],counts,width=np.diff(edges),align='edge',color=colors,edgecolor='#222222',linewidth=.28,alpha=.9)
    ax.set_xlim(0,edges[-1]*1.01);ax.set_ylim(bottom=0);ax.set_xlabel('Delivery burden (MET h)')
    ax.set_ylabel('Number of couriers' if panel=='e' else 'Number of orders')
    ax.xaxis.set_major_locator(MaxNLocator(5));fmt=ScalarFormatter(useMathText=True);fmt.set_powerlimits((4,4));fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt);ax.yaxis.get_offset_text().set_x(1);ax.yaxis.get_offset_text().set_ha('right')

# Load minutes
def load_minutes(work,c):
    z=np.load(work/'minute_groups.npz',allow_pickle=False)
    return minute_sources(z['counts'],z['sums'],z['upstairs'],z['orders'],c.get('start_minute',420))

# Render figure panels
def run(c,work,out,panels='abcdef',combined=True):
    style();start=c.get('start_minute',420);end=c.get('end_minute',1140);audit={}
    data,ag,bg=load_minutes(work,c) if ('c' in panels or 'd' in panels) else (None,None,None)
    city=pd.read_csv(work/'a_city_counts.csv') if 'a' in panels else None
    b=prepare_b(c) if 'b' in panels else None
    if b is not None:
        audit['b']=b[-1];write_json(out/'b_spatial_audit.json',{'counts':b[-1],'mapping':'One point to the interior anchor of smallest containing positive-burden AOI; unmatched points retained without links; not a trajectory'})
    if data is not None:
        data.to_csv(out/'cd_minute_means.csv',index=False)
        audit['c']={'compressed_interval_seconds':[18,27],'points_in_compressed_interval':int(((ag>18)&(ag<27)).sum()),'clipped_points':int(((ag<0)|(ag>55)).sum())}
        audit['d']={'clipped_points':int(((bg<.023992688302870546)|(bg>.028)).sum())}
    hist={p:pd.read_csv(work/f'{p}_histogram.csv') for p in 'ef' if p in panels}
    # Draw
    def draw(fig,ax,p):
        if p=='a':plot_a(fig,ax,c,city)
        elif p=='b':plot_b(ax,b)
        elif p=='c':return plot_c(ax,data,ag,start,end)
        elif p=='d':return plot_d(ax,data,bg,start,end)
        else:plot_hist(ax,hist[p],p)
    for p in panels:
        fig=plt.figure(figsize=(12,8) if p in 'ab' else (12,4))
        ax=fig.add_axes([.11,.23 if p=='a' else .19,.80,.68 if p=='a' else .66])
        m=draw(fig,ax,p)
        if p in 'cd':
            cax=fig.add_axes([.93,.19,.014,.66]);cb=fig.colorbar(m,cax=cax);cb.set_label('Order volume per minute')
        save(fig,out,'Figure1'+p);plt.close(fig)
    if combined and set(panels)==set('abcdef'):
        fig=plt.figure(figsize=(12,13))
        positions={'a':[.04,.665,.53,.30],'b':[.60,.645,.37,.34],
          'c':[.10,.445,.83,.165],'d':[.10,.248,.83,.165],
          'e':[.10,.061,.365,.139],'f':[.585,.061,.365,.139]}
        for p,pos in positions.items():
            ax=fig.add_axes(pos);m=draw(fig,ax,p)
            fig.text(pos[0]-.055,pos[1]+pos[3]+.008,p,fontweight='bold',fontsize=16)
            if p=='d':
                cax=fig.add_axes([.94,.333,.009,.167]);cb=fig.colorbar(m,cax=cax)
                ticks=[t for t in [0,20000,40000,60000] if t<=m.norm.vmax];cb.set_ticks(ticks);cb.set_ticklabels([str(t//1000) for t in ticks]);cb.set_label('Order volume (thousands)')
        save(fig,out,'Figure1');plt.close(fig)
    write_json(out/'plot_audit.json',audit)
