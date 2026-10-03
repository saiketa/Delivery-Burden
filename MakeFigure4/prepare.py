# Activity calculations
import importlib.util
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Shared label readers
def shared_readers():
    folder=Path(__file__).resolve().parents[1]/'MakeFigure2'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('figure2_readers',folder/'prepare.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Aggregate activity seconds
def load_durations(db,files,readers):
    if not files or len(set(files))!=len(files):raise ValueError('Invalid behavior files')
    seconds=[a+'_seconds' for a in readers.ACTIVITIES]
    keys=['aoi_source_file','aoiId','userid']
    for filename in files:
        for ch in pd.read_csv(filename,usecols=keys+seconds,dtype=str,chunksize=100000):
            for key in keys:ch[key]=readers.normalize(ch[key])
            if ch[keys].eq('').any().any():raise ValueError('Missing behavior keys')
            values=ch[seconds].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values<0).any():
                raise ValueError('Invalid activity seconds')
            ch['stationary']=values[:,0];ch['walking']=values[:,1]
            ch['driving']=values[:,2];ch['stairs']=values[:,3]+values[:,4]
            d=ch.groupby(keys[:2],sort=False)[['stationary','walking','driving','stairs']].sum().reset_index()
            db.executemany('''INSERT INTO durations VALUES(?,?,?,?,?,?)
              ON CONFLICT(source,aoi) DO UPDATE SET
              stationary=durations.stationary+excluded.stationary,
              walking=durations.walking+excluded.walking,
              driving=durations.driving+excluded.driving,
              stairs=durations.stairs+excluded.stairs''',d.itertuples(index=False,name=None))
            db.commit()

# City-equal bootstrap intervals
def summarize(cities):
    rng=np.random.default_rng(20240830)
    records=[]
    for metric,denominator,unit in [('per_courier','courier_count',60),
                                   ('per_package','order_count',1),('per_AOI','aoi_count',60)]:
        for (tier,cls),group in cities.groupby(['tier','cls'],sort=False):
            for activity in ['stationary','walking','driving','stairs']:
                values=(group[activity]/group[denominator]/unit).to_numpy()
                if not np.isfinite(values).all():raise ValueError('Invalid city activity ratios')
                means=np.empty(10000)
                for start in range(0,10000,250):
                    means[start:start+250]=rng.choice(values,(250,len(values)),replace=True).mean(axis=1)
                low,high=np.percentile(means,[2.5,97.5])
                records.append([metric,activity,tier,cls,len(values),values.mean(),low,high])
    return pd.DataFrame(records,columns=['metric','behavior','tier','cls','n_cities','mean','ci95_low','ci95_high'])

# Prepare city activity ratios
def prepare_sources(inputs,work):
    target=work/'activity.sqlite3'
    if target.exists():raise FileExistsError(target)
    readers=shared_readers()
    with sqlite3.connect(target) as db:
        db.executescript('''CREATE TABLE meta(source TEXT,aoi TEXT,city TEXT,tier TEXT,cls TEXT,PRIMARY KEY(source,aoi));
          CREATE TABLE courier_labels(userid TEXT,city TEXT,tier TEXT,cls TEXT,PRIMARY KEY(userid,city,tier,cls));
          CREATE TABLE matched(userid TEXT,city TEXT,tier TEXT,cls TEXT,n INTEGER,PRIMARY KEY(userid,city,tier,cls));
          CREATE TABLE durations(source TEXT,aoi TEXT,stationary REAL,walking REAL,driving REAL,stairs REAL,PRIMARY KEY(source,aoi));''')
        readers.load_aois(db,inputs['aoi_csv'])
        readers.load_couriers(db,inputs['courier_csv'])
        readers.load_matched_orders(db,inputs['orders_with_aoi'])
        load_durations(db,inputs['behavior_files'],readers)
        missing=db.execute('''SELECT COUNT(*) FROM durations d LEFT JOIN meta m USING(source,aoi)
          WHERE m.aoi IS NULL OR m.tier IS NULL OR m.cls NOT IN ('urban','rural') OR m.cls IS NULL''').fetchone()[0]
        if missing:raise ValueError(f'Missing AOI geographical labels: {missing}')
        cities=pd.read_sql_query('''WITH totals AS (
          SELECT city,tier,cls,COUNT(*) aoi_count,SUM(stationary) stationary,
          SUM(walking) walking,SUM(driving) driving,SUM(stairs) stairs
          FROM durations JOIN meta USING(source,aoi)
          WHERE stationary+walking+driving+stairs>0 GROUP BY city,tier,cls),
          people AS (SELECT city,tier,cls,COUNT(DISTINCT userid) courier_count
          FROM courier_labels GROUP BY city,tier,cls),
          orders AS (SELECT city,tier,cls,SUM(n) order_count FROM matched GROUP BY city,tier,cls)
          SELECT t.*,p.courier_count,o.order_count FROM totals t
          LEFT JOIN people p USING(city,tier,cls) LEFT JOIN orders o USING(city,tier,cls)''',db)
    if cities.empty:raise ValueError('No classified city activity totals')
    denominators=cities[['courier_count','order_count','aoi_count']]
    if denominators.isna().any().any() or (denominators<=0).any().any():
        raise ValueError('Missing or nonpositive city denominators')
    cities.to_csv(work/'city_activity_totals.csv',index=False)
    summarize(cities).to_csv(work/'activity_statistics.csv',index=False)
