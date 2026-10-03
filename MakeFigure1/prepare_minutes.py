# Minute calculations
import sqlite3
import numpy as np
import pandas as pd
from common import group_id, normalize, path, minute_sources, write_json

# Calculate minute activity and burden
def run(c, out):
    date = c.get('date','2024-08-30')
    start, end = c.get('start_minute',420), c.get('end_minute',1140)
    db = sqlite3.connect(path(c,'minute_db').resolve().as_uri()+'?mode=ro', uri=True)
    db.create_function('courier_group',1,lambda uid:group_id(uid,date),deterministic=True)
    lower=f'{date} {start//60:02}:{start%60:02}:00'; upper=f'{date} {end//60:02}:{end%60:02}:00'
    duplicates=db.execute('''SELECT 1 FROM agg
      WHERE minute_start>=? AND minute_start<? GROUP BY userid,minute_start HAVING COUNT(*)>1 LIMIT 1''',(lower,upper)).fetchone()
    if duplicates: raise ValueError('agg must contain exactly one row per courier and minute')
    q='''SELECT minute_start,courier_group(userid) g,COUNT(*) n,SUM(still) still,SUM(walking) walking,
      SUM(driving) driving,SUM(upstairs) upstairs,SUM(downstairs) downstairs
      FROM agg WHERE minute_start>=? AND minute_start<?
      GROUP BY minute_start,g'''
    v=pd.read_sql_query(q,db,params=(lower,upper)); db.close()
    counts=np.zeros((end-start,800),dtype='int64'); sums=np.zeros((end-start,800,4)); up=np.zeros_like(counts,dtype=float)
    m=v.minute_start.str.slice(11,13).astype(int)*60+v.minute_start.str.slice(14,16).astype(int)-start; g=v.g.to_numpy()
    counts[m,g]=v.n; sums[m,g,0]=v.still; sums[m,g,1]=v.walking; sums[m,g,2]=v.driving
    sums[m,g,3]=v.upstairs+v.downstairs; up[m,g]=v.upstairs
    orders=np.zeros(1440,dtype='int64'); audit=[]
    for filename in c['order_files']:
        f=path({**c,'_file':filename},'_file'); before=int(orders.sum())
        for chunk in pd.read_csv(f,usecols=['state','create_time'],dtype=str,chunksize=300000):
            state=normalize(chunk.state)
            ok=state.isin(['150','-640'])&chunk.create_time.str.startswith(date+' ',na=False)
            tm=pd.to_datetime(chunk.loc[ok,'create_time'],format='mixed',errors='raise')
            orders+=np.bincount(tm.dt.hour*60+tm.dt.minute,minlength=1440)
        audit.append({'input_index':len(audit),'order_events':int(orders.sum())-before})
        print('Order input',len(audit),'complete',flush=True)
    np.savez_compressed(out/'minute_groups.npz',counts=counts,sums=sums,upstairs=up,orders=orders)
    frame,_,_=minute_sources(counts,sums,up,orders,start);frame.to_csv(out/'minute_means.csv',index=False)
    write_json(out/'minutes_audit.json',{'observed_courier_minutes':int(counts.sum()),
        'order_events_all_day':int(orders.sum()),'order_events_display_window':int(frame.order_events.sum()),
        'date':date,'start_minute':start,'end_minute':end,'order_inputs':audit})
