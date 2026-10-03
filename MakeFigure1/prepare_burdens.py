# Daily burden calculations
import sqlite3
import numpy as np
import pandas as pd
from common import ACTIVITIES, burden, normalize, path, histogram, write_json

# Calculate courier and order burdens
def run(c,out):
    if len(c['behavior_files']) != c.get('expected_behavior_parts',20):
        raise ValueError('Specify every disjoint behavior partition; default expected count is 20')
    work=out/'burden_work.sqlite3'
    if work.exists(): raise FileExistsError('Use a fresh work directory; refusing to mix prior runs')
    db=sqlite3.connect(work)
    db.executescript('''CREATE TABLE aoi(source TEXT,aoi TEXT,burden REAL,PRIMARY KEY(source,aoi));
      CREATE TABLE courier(userid TEXT PRIMARY KEY,burden REAL);
      CREATE TABLE orders(source TEXT,aoi TEXT,n INTEGER,PRIMARY KEY(source,aoi));
      CREATE TEMP TABLE stage(source TEXT,aoi TEXT,userid TEXT,burden REAL);''')
    stats={'behavior_rows':0,'missing_seconds_imputed_zero':0,'order_rows':0,'excluded_missing_AOI_key_rows':0}
    columns=['aoi_source_file','aoiId','userid']+[a+'_seconds' for a in ACTIVITIES]
    if len(set(c['behavior_files'])) != len(c['behavior_files']): raise ValueError('Repeated behavior input')
    for filename in c['behavior_files']:
        f=path({**c,'_file':filename},'_file')
        for ch in pd.read_csv(f,usecols=columns,dtype=str,chunksize=100000):
            stats['behavior_rows']+=len(ch)
            seconds=ch[[a+'_seconds' for a in ACTIVITIES]].apply(pd.to_numeric,errors='coerce')
            stats['missing_seconds_imputed_zero']+=int(seconds.isna().sum().sum())
            values=burden(seconds.fillna(0).to_numpy())
            ch['burden']=values
            for col in ['aoi_source_file','aoiId','userid']: ch[col]=normalize(ch[col])
            stats['excluded_missing_AOI_key_rows']+=int((ch.aoiId.eq('')|ch.aoi_source_file.eq('')).sum())
            ch=ch.loc[(ch.burden>0)&ch.aoiId.ne('')&ch.aoi_source_file.ne('')]
            db.execute('DELETE FROM stage')
            db.executemany('INSERT INTO stage VALUES(?,?,?,?)',ch[['aoi_source_file','aoiId','userid','burden']].itertuples(index=False,name=None))
            db.execute('''INSERT INTO courier SELECT userid,SUM(burden) FROM stage
              GROUP BY userid ON CONFLICT(userid) DO UPDATE SET burden=courier.burden+excluded.burden''')
            db.execute('''INSERT INTO aoi SELECT source,aoi,SUM(burden) FROM stage GROUP BY source,aoi
              ON CONFLICT(source,aoi) DO UPDATE SET burden=aoi.burden+excluded.burden''')
            db.commit()
        print('Daily behavior input complete',flush=True)
    aoi_frame=pd.read_sql_query('SELECT source aoi_source_file,aoi aoiId,burden delivery_load_total FROM aoi ORDER BY source,aoi',db)
    aoi_frame.to_csv(out/'aoi_burdens.csv',index=False)
    stats['positive_AOIs']=len(aoi_frame)
    for ch in pd.read_csv(path(c,'classified_orders_csv'),usecols=['aoi_source_file','aoiId'],dtype=str,chunksize=300000):
        stats['order_rows']+=len(ch)
        for col in ['aoi_source_file','aoiId']:ch[col]=normalize(ch[col])
        ch=ch.loc[ch.aoiId.ne('')&ch.aoi_source_file.ne('')]
        v=ch.groupby(['aoi_source_file','aoiId']).size().reset_index(name='n')
        db.executemany('INSERT INTO orders VALUES(?,?,?) ON CONFLICT(source,aoi) DO UPDATE SET n=orders.n+excluded.n',v.itertuples(index=False,name=None));db.commit()
    e=pd.read_sql_query('SELECT userid,burden FROM courier WHERE burden>0',db)
    e.to_csv(out/'courier_burdens.csv',index=False)
    f=pd.read_sql_query('SELECT a.burden/o.n burden,o.n FROM aoi a JOIN orders o USING(source,aoi) WHERE o.n>0',db)
    for panel,d,w in [('e',e,None),('f',f,f.n.to_numpy())]:
        bins,audit=histogram(d.burden.to_numpy(),w)
        bins.to_csv(out/f'{panel}_histogram.csv',index=False);stats[panel]=audit
    stats['total_burden_MET_h']=float(e.burden.sum())
    stats['positive_burden_couriers']=len(e)
    stats['orders_with_positive_burden_AOI']=int(f.n.sum())
    stats['positive_AOIs_without_orders']=int(len(aoi_frame)-len(f))
    db.close();write_json(out/'burdens_audit.json',stats)
