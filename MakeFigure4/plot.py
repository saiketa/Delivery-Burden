# Activity comparison figure
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Figure style
TIERS=['Tier 1','New Tier 1','Tier 2','Tier 3','Tier 4','Tier 5']
COLORS={'urban':'#0878B9','rural':'#D9164B'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':23,'axes.labelsize':23,'axes.titlesize':23,'xtick.labelsize':23,
    'ytick.labelsize':23,'legend.fontsize':23,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':1.0,'legend.frameon':False})

# Draw twelve comparisons
def make_figure(work,output):
    data=pd.read_csv(work/'activity_statistics.csv')
    if data.duplicated(['metric','behavior','tier','cls']).any():
        raise ValueError('Duplicate activity statistics')
    fig,axes=plt.subplots(3,4,figsize=(22,16))
    fig.subplots_adjust(left=.095,right=.985,bottom=.10,top=.92,wspace=.36,hspace=.60)
    rows=[('per_courier','courier (min)'),('per_package','package (s)'),('per_AOI','AOI (min)')]
    columns=[('stationary','Stationary'),('walking','Walking'),('driving','Driving'),('stairs','Stairs')]
    for r,(metric,unit) in enumerate(rows):
        for c,(activity,title) in enumerate(columns):
            ax=axes[r,c]
            sub=data.loc[data.metric.eq(metric)&data.behavior.eq(activity)]
            for cls,offset,marker in [('urban',-.13,'o'),('rural',.13,'s')]:
                d=sub.loc[sub.cls.eq(cls)].set_index('tier').reindex(TIERS)
                values=d[['mean','ci95_low','ci95_high']].to_numpy(dtype=float)
                if not np.isfinite(values).all():raise ValueError('Missing tier activity statistics')
                mean,low,high=values.T
                if (low>mean).any() or (high<mean).any():raise ValueError('Invalid confidence interval')
                ax.errorbar(np.arange(6)+offset,mean,yerr=[mean-low,high-mean],
                    fmt=marker,markersize=8.5,color=COLORS[cls],ecolor=COLORS[cls],
                    elinewidth=1.6,capsize=5,capthick=1.4,markeredgecolor='white',
                    markeredgewidth=.7,zorder=3)
            low,high=sub.ci95_low.min(),sub.ci95_high.max()
            pad=max(high-low,abs(high)*.05,1e-6)*.12
            ax.set_ylim(max(0,low-pad),high+pad)
            ax.set_xlim(-.55,5.55)
            ax.set_xticks(range(6),TIERS,rotation=30,ha='right',rotation_mode='anchor')
            ax.set_title(title,pad=12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(direction='out',length=4,width=1,pad=5)
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_color('#333333')
            if c==0:ax.set_ylabel('Activity duration\nper '+unit,labelpad=16)
        pos=axes[r,0].get_position()
        fig.text(.022,pos.y1+.018,'abc'[r],fontsize=30,fontweight='bold',va='bottom')
    handles=[Line2D([],[],marker=m,linestyle='none',color=COLORS[k],markersize=10,label=k.title())
             for k,m in [('urban','o'),('rural','s')]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.985),ncol=2,
               columnspacing=2.3,handletextpad=.4)
    fig.text(.54,.025,'City tier',ha='center',fontsize=23)
    fig.savefig(output/'Four_activity_duration_3x4_20240830.pdf',facecolor='white')
    fig.savefig(output/'Four_activity_duration_3x4_20240830.svg',facecolor='white')
    fig.savefig(output/'Four_activity_duration_3x4_20240830.png',dpi=300,facecolor='white')
    plt.close(fig)
