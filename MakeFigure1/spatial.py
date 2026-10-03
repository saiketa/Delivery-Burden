# Spatial panels
import json
import colorsys
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from common import path, normalize, write_json

# Polygons
def polygons(g):
    if g.geom_type=='Polygon': yield g
    elif hasattr(g,'geoms'):
        for part in g.geoms: yield from polygons(part)

# Features
def features(p):
    obj=json.loads(p.read_text(encoding='utf-8'))
    return obj['features']

# Coordinate conversion
def wgs84_to_gcj02(lng,lat):
    lng=np.asarray(lng,float);lat=np.asarray(lat,float);x=lng-105;y=lat-35;pi=np.pi
    dlat=-100+2*x+3*y+.2*y*y+.1*x*y+.2*np.sqrt(np.abs(x))
    dlat+=(20*np.sin(6*x*pi)+20*np.sin(2*x*pi))*2/3
    dlat+=(20*np.sin(y*pi)+40*np.sin(y*pi/3))*2/3
    dlat+=(160*np.sin(y*pi/12)+320*np.sin(y*pi/30))*2/3
    dlng=300+x+2*y+.1*x*x+.1*x*y+.1*np.sqrt(np.abs(x))
    dlng+=(20*np.sin(6*x*pi)+20*np.sin(2*x*pi))*2/3
    dlng+=(20*np.sin(x*pi)+40*np.sin(x*pi/3))*2/3
    dlng+=(150*np.sin(x*pi/12)+300*np.sin(x*pi/30))*2/3
    a=6378245.;ee=.00669342162296594323;rad=lat*pi/180;magic=1-ee*np.sin(rad)**2
    dlat=dlat*180/((a*(1-ee))/(magic*np.sqrt(magic))*pi)
    dlng=dlng*180/(a/np.sqrt(magic)*np.cos(rad)*pi)
    outside=(lng<72.004)|(lng>137.8347)|(lat<.8293)|(lat>55.8271)
    return np.c_[np.where(outside,lng,lng+dlng),np.where(outside,lat,lat+dlat)]

# Coordinates
def coordinates(d,x,y,datum,target):
    v=d[[x,y]].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(v).all(): raise ValueError('Nonfinite spatial coordinates')
    if datum==target: return v
    if datum=='WGS84' and target=='GCJ02': return wgs84_to_gcj02(v[:,0],v[:,1])
    raise ValueError(f'Unsupported datum transform: {datum} to {target}')

# Prepare national distribution
def prepare_a(c,out):
    fs=features(path(c,'city_boundaries'))
    names=[f['properties'][c.get('city_name_property','市')] for f in fs]
    geoms=[shapely.make_valid(shape(f['geometry'])) for f in fs];tree=shapely.STRtree(geoms)
    couriers={n:0 for n in names};aois={n:0 for n in names};seen=set();unmatched=0
    for ch in pd.read_csv(path(c,'national_courier_points'),usecols=['userid','lng','lat'],dtype={'userid':str},chunksize=20000):
        if normalize(ch.userid).isin(seen).any() or normalize(ch.userid).duplicated().any():
            raise ValueError('Courier unique-point CSV contains repeated IDs')
        seen.update(normalize(ch.userid));xy=coordinates(ch,'lng','lat',c['national_point_datum'],c['city_boundary_datum'])
        for p in shapely.points(xy):
            hits=sorted({names[i] for i in tree.query(p,predicate='intersects')})
            if len(hits)==1: couriers[hits[0]]+=1
            else: unmatched+=1
    aoi_ids={}
    for ch in pd.read_csv(path(c,'aoi_csv'),usecols=['aoi_source_file','aoiId'],dtype=str,chunksize=100000):
        for source,part in ch.groupby('aoi_source_file'):
            aoi_ids.setdefault(source,set()).update(part.aoiId.dropna())
    for source,ids in aoi_ids.items():
        city=source.split('_AOI')[0]
        if city not in aois:raise ValueError('AOI city not in boundary data: '+city)
        aois[city]+=len(ids)
    rows=[]
    lookup={n:g for n,g in zip(names,geoms)}
    for n in sorted(lookup):
        p=max(polygons(lookup[n]),key=lambda p:p.area).representative_point()
        rows.append([n,couriers[n],aois[n],p.x,p.y])
    pd.DataFrame(rows,columns=['city','courier_count','aoi_count','lng','lat']).to_csv(out/'a_city_counts.csv',index=False)
    write_json(out/'a_audit.json',{'unique_couriers':len(seen),'unmatched_couriers':unmatched,'AOIs':sum(aois.values())})

# Panel a: National distribution
def plot_a(fig,ax,c,data):
    fs=features(path(c,'city_boundaries'));geoms=[shapely.make_valid(shape(f['geometry'])) for f in fs]
    data=data.sort_values('aoi_count',ascending=False);maximum=data.aoi_count.max()
    cmap=LinearSegmentedColormap.from_list('couriers',['#deeeea','#82b5a6','#347b70','#084c48'])
    norm=LogNorm(1,max(2,float(data.courier_count.max())))
    # Draw
    def draw(a,limits,inset=False):
        poly=[np.asarray(p.exterior.coords) for g in geoms for p in polygons(shapely.simplify(g,.025,preserve_topology=True))]
        a.add_collection(PolyCollection(poly,facecolors='#fafbf9',edgecolors='#c6ceca',linewidths=.35))
        marks=data[(data.aoi_count>0)&(data.courier_count>0)]
        rgba=cmap(norm(marks.courier_count.to_numpy()));highlight=marks.city.isin(['北京市','上海市','乌鲁木齐市','拉萨市']).to_numpy()
        for j in np.where(highlight)[0]:
            _,light,sat=colorsys.rgb_to_hls(*rgba[j,:3]);rgba[j,:3]=colorsys.hls_to_rgb(0,light,sat)
        size=marks.aoi_count/maximum*220
        a.scatter(marks.lng,marks.lat,s=size,c=rgba,edgecolors='white',linewidths=.3,zorder=3)
        a.scatter(marks.lng.to_numpy()[highlight],marks.lat.to_numpy()[highlight],s=size.to_numpy()[highlight],c=rgba[highlight],edgecolors='white',linewidths=.3,zorder=4)
        a.set(xlim=limits[:2],ylim=limits[2:]);a.set_aspect(1/np.cos(np.deg2rad(35)));a.set_xticks([]);a.set_yticks([])
        for sp in a.spines.values():sp.set_visible(inset)
    draw(ax,(73,136,17,54))
    for city,label,offset in [('北京市','Beijing',(6,7)),('上海市','Shanghai',(8,0)),('乌鲁木齐市','Urumqi',(5,-8)),('拉萨市','Lhasa',(-7,10))]:
        v=data.loc[data.city.eq(city)]
        if len(v):ax.annotate(label,(v.iloc[0].lng,v.iloc[0].lat),xytext=offset,textcoords='offset points',fontsize=9,zorder=6)
    sea=ax.inset_axes([.77,.03,.19,.29]);sea.set_facecolor('#e5e9e8');draw(sea,(105,125,2,25),True)
    cax=ax.inset_axes([.48,-.12,.5,.027]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    cb.ax.set_title('Number of couriers',fontsize=9,pad=4)
    ticks=[t for t in [1,10,100,1000,5000] if t<=norm.vmax];cb.set_ticks(ticks);cb.set_ticklabels([f'{v:,}' for v in ticks]);cb.ax.tick_params(labelsize=8)
    leg=ax.inset_axes([.02,-.17,.43,.11]);leg.axis('off');leg.set(xlim=(0,1),ylim=(0,1));leg.text(.5,1.05,'Number of AOIs',ha='center',fontsize=9)
    for x,v in zip([.1,.36,.63,.90],[1000,10000,50000,100000]):
        leg.scatter(x,.60,s=v/maximum*220,color='#82b5a6');leg.text(x,.1,f'{v:,}',ha='center',fontsize=8)

# Prepare spatial mapping
def prepare_b(c):
    district=shapely.make_valid(shapely.union_all([shape(f['geometry']) for f in features(path(c,'district_boundary'))]))
    parts=[]
    for ch in pd.read_csv(path(c,'aoi_csv'),usecols=['aoi_source_file','aoiId','aoiWKT'],dtype=str,chunksize=100000):
        parts.append(ch.loc[ch.aoi_source_file.str.startswith('上海市_AOI',na=False)])
    a=pd.concat(parts,ignore_index=True)
    full=shapely.make_valid(shapely.from_wkt(a.aoiWKT.to_numpy()))
    keep=shapely.intersects(full,district);a=a.loc[keep].copy();full=full[keep];g=shapely.intersection(full,district)
    keep=shapely.area(g)>0;a=a.loc[keep].reset_index(drop=True);g=g[keep];full=full[keep]
    points=shapely.point_on_surface(g);anchors=np.c_[shapely.get_x(points),shapely.get_y(points)]
    tree=shapely.STRtree(g);areas=shapely.area(full);layers=[];audit=[]
    if c['aoi_datum']!=c['district_datum']:raise ValueError('AOI polygons must already be in district datum')
    for kind,key,cols,datum,uid in [('Couriers','district_courier_points',('lng','lat'),c['district_courier_datum'],'userid'),('Orders','district_order_points',('location_lng','location_lat'),c['district_order_datum'],'operator_user_id')]:
        pieces=[];seen=set();invalid=0
        for d in pd.read_csv(path(c,key),usecols=[uid,*cols],dtype={uid:str},chunksize=200000):
            if kind=='Couriers':
                if normalize(d[uid]).duplicated().any() or normalize(d[uid]).isin(seen).any():raise ValueError('Repeated courier unique points')
                seen.update(normalize(d[uid]))
            v=d[list(cols)].apply(pd.to_numeric,errors='coerce');valid=np.isfinite(v).all(axis=1);invalid+=int((~valid).sum());d=d.loc[valid].copy()

            x0,y0,x1,y1=district.bounds
            near=v.loc[valid,cols[0]].between(x0-.02,x1+.02)&v.loc[valid,cols[1]].between(y0-.02,y1+.02)
            d=d.loc[near]
            xy=coordinates(d,*cols,datum,c['district_datum']);pieces.append(xy[shapely.intersects(shapely.points(xy),district)])
        xy=np.concatenate(pieces) if pieces else np.empty((0,2))
        pairs=tree.query(shapely.points(xy),predicate='intersects')
        order=np.lexsort((pairs[1],areas[pairs[1]],pairs[0]));pairs=pairs[:,order]
        first=np.r_[True,np.diff(pairs[0])!=0] if pairs.shape[1] else np.zeros(0,dtype=bool)
        pi,ai=pairs[:,first];layers.append((kind,xy,pi,ai))
        audit.append({'layer':kind,'points':len(xy),'matched':len(pi),'unmatched':len(xy)-len(pi),'AOIs':len(np.unique(ai)),'invalid_coordinates':invalid})
    return district,g,anchors,layers,audit

# Panel b: Spatial mapping
def plot_b(ax,prepared):
    district,geoms,anchors,layers,audit=prepared
    xmin,ymin,xmax,ymax=district.bounds
    scale=28/((ymax-ymin)*111+.32*(xmax-xmin)*95.2)
    # Proj
    def proj(xy,z=0):
        xy=np.asarray(xy);x=(xy[:,0]-xmin)*95.2;y=(xy[:,1]-ymin)*111
        return np.c_[((ymax-ymin)*111-y+.32*x)*scale,.40*x*scale+z]
    border=[np.asarray(p.exterior.coords) for p in polygons(district)]
    height=max(proj(b)[:,1].max() for b in border);gap=height+4.914
    for kind,xy,pi,ai in layers:
        top=kind=='Couriers';z=2*gap if top else 0
        lines=np.stack([proj(xy[pi],z),proj(anchors[ai],gap)],axis=1)
        ax.add_collection(LineCollection(lines,colors='#777777',linewidths=.5 if top else .1,alpha=.25 if top else .06,zorder=1,rasterized=True))
        q=proj(xy,z);ax.scatter(*q.T,s=2 if top else .45,color='#9870CC' if top else '#DCA185',alpha=.8 if top else .4,edgecolors='none',zorder=3,rasterized=True)
    outlines=[proj(p.exterior.coords,gap) for geom in geoms for p in polygons(geom)]
    ax.add_collection(LineCollection(outlines,colors='#468C69',linewidths=.25,alpha=.55,zorder=4))
    for text,z,color in [('Couriers',2*gap,'#9870CC'),('AOIs',gap,'#468C69'),('Orders',0,'#DCA185')]:
        ax.add_collection(LineCollection([proj(b,z) for b in border],colors=color,linewidths=.5,zorder=5))
        ax.text(-1,z+height*.5,text,ha='right',va='center',fontsize=10)
    ax.set(xlim=(-6,29),ylim=(-1,2*gap+height+1));ax.set_aspect('equal');ax.axis('off')
