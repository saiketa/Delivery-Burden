# Two-period calculations
import sqlite3
import numpy as np
import pandas as pd
from model import ACTIVITIES, burden, normalize, city_name, tier_label, write_json

# AOI attributes
def load_aois(db,filename):
    columns=['aoi_source_file','aoiId','tier','rural_urban']
    for ch in pd.read_csv(filename,usecols=columns,dtype=str,chunksize=100000):
        ch['source']=normalize(ch.aoi_source_file);ch['aoi']=normalize(ch.aoiId)
        if ch.source.eq('').any() or ch.aoi.eq('').any():raise ValueError('Missing AOI key')
        ch['city']=ch.source.map(city_name);ch['tier']=ch.tier.map(tier_label)
        ch['cls']=normalize(ch.rural_urban).str.lower()
        ch.loc[~ch.cls.isin(['urban','rural']),'cls']=None
        records=ch[['source','aoi','city','tier','cls']].astype(object)
        records=records.where(records.notna(),None)
        db.executemany('INSERT INTO meta VALUES(?,?,?,?,?)',records.itertuples(index=False,name=None))
        db.commit()
    conflicting=db.execute('SELECT city FROM meta GROUP BY city HAVING COUNT(DISTINCT tier)>1 LIMIT 1').fetchone()
    if conflicting:raise ValueError('Conflicting city tiers')

# Behavior aggregation
def load_behavior(db,files):
    if len(set(files))!=len(files) or not files:raise ValueError('Invalid behavior file list')
    stats={'behavior_rows':0,'missing_seconds':0,'invalid_keys':0}
    cols=['aoi_source_file','aoiId','userid']+[a+'_seconds' for a in ACTIVITIES]
    sql='''INSERT INTO behavior VALUES(?,?,?,?,?) ON CONFLICT(source,aoi,userid)
      DO UPDATE SET seconds=behavior.seconds+excluded.seconds,burden=behavior.burden+excluded.burden'''
    for filename in files:
        for ch in pd.read_csv(filename,usecols=cols,dtype=str,chunksize=100000):
            stats['behavior_rows']+=len(ch)
            for col in ['aoi_source_file','aoiId','userid']:ch[col]=normalize(ch[col])
            sec=ch[[a+'_seconds' for a in ACTIVITIES]].apply(pd.to_numeric,errors='coerce')
            stats['missing_seconds']+=int(sec.isna().sum().sum());sec=sec.fillna(0).to_numpy()
            ch['seconds']=sec.sum(axis=1);ch['burden']=burden(sec)
            valid=ch.aoi_source_file.ne('')&ch.aoiId.ne('')&ch.userid.ne('')
            stats['invalid_keys']+=int((~valid).sum())
            db.executemany(sql,ch.loc[valid,['aoi_source_file','aoiId','userid','seconds','burden']].itertuples(index=False,name=None));db.commit()
    return stats

# Supplied geographical labels
def supplied_labels(ch):
    ch['city']=normalize(ch.city)
    ch['tier']=ch.tier.map(tier_label)
    ch['cls']=normalize(ch.rural_urban).str.lower()
    if ch.city.eq('').any() or ch.tier.isna().any() or not ch.cls.isin(['urban','rural']).all():
        raise ValueError('Missing or invalid supplied geographical labels')
    return ch

# Supplied courier memberships
def load_couriers(db,filename):
    cols=['userid','city','tier','rural_urban']
    for ch in pd.read_csv(filename,usecols=cols,dtype=str,chunksize=100000):
        ch=supplied_labels(ch);ch['userid']=normalize(ch.userid)
        if ch.userid.eq('').any():raise ValueError('Missing courier ID')
        db.executemany('INSERT OR IGNORE INTO courier_labels VALUES(?,?,?,?)',
            ch[['userid','city','tier','cls']].itertuples(index=False,name=None))
        db.commit()
    conflict=db.execute('''SELECT userid FROM courier_labels GROUP BY userid
      HAVING COUNT(DISTINCT city)>1 OR COUNT(DISTINCT tier)>1 LIMIT 1''').fetchone()
    if conflict:raise ValueError('Supply one city and tier per courier; both rural/urban memberships are allowed')

# Preclassified order aggregation
def load_matched_orders(db,filename):
    stats={'matched_input_rows':0,'unmatched_order_rows':0}
    cols=['operator_user_id','city','tier','rural_urban']
    for ch in pd.read_csv(filename,usecols=cols,dtype=str,chunksize=200000):
        stats['matched_input_rows']+=len(ch)
        ch=supplied_labels(ch);ch['operator_user_id']=normalize(ch.operator_user_id)
        if ch.operator_user_id.eq('').any():raise ValueError('Missing order courier ID')
        d=ch.groupby(['operator_user_id','city','tier','cls'],sort=False).size().reset_index(name='n')
        db.executemany('''INSERT INTO matched VALUES(?,?,?,?,?) ON CONFLICT(userid,city,tier,cls)
          DO UPDATE SET n=matched.n+excluded.n''',d.itertuples(index=False,name=None));db.commit()
    return stats

# Courier order totals
def load_orders(db,files,date):
    day=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    for filename in files:
        for ch in pd.read_csv(filename,usecols=['operator_user_id','state','create_time'],dtype=str,chunksize=200000):
            uid=normalize(ch.operator_user_id);state=normalize(ch.state)
            valid=uid.ne('')&state.isin(['150','-640'])&ch.create_time.str.startswith(day+' ',na=False)
            d=uid.loc[valid].value_counts()
            db.executemany('INSERT INTO courier_orders VALUES(?,?) ON CONFLICT(userid) DO UPDATE SET n=courier_orders.n+excluded.n',[(u,int(n)) for u,n in d.items()]);db.commit()

# Prepare period
def prepare_date(inputs,date,work):
    target=work/f'{date}.sqlite3'
    if target.exists():raise FileExistsError(target)
    db=sqlite3.connect(target)
    db.executescript('''CREATE TABLE meta(source TEXT,aoi TEXT,city TEXT,tier TEXT,cls TEXT,PRIMARY KEY(source,aoi));
      CREATE TABLE behavior(source TEXT,aoi TEXT,userid TEXT,seconds REAL,burden REAL,PRIMARY KEY(source,aoi,userid));
      CREATE TABLE matched(userid TEXT,city TEXT,tier TEXT,cls TEXT,n INTEGER,PRIMARY KEY(userid,city,tier,cls));
      CREATE TABLE courier_labels(userid TEXT,city TEXT,tier TEXT,cls TEXT,PRIMARY KEY(userid,city,tier,cls));
      CREATE TABLE courier_orders(userid TEXT PRIMARY KEY,n INTEGER);''')
    load_aois(db,inputs['aoi_csv']);stats=load_behavior(db,inputs['behavior_files'])
    load_couriers(db,inputs['courier_csv'])
    stats.update(load_matched_orders(db,inputs['orders_with_aoi']))
    load_orders(db,inputs['order_files'],date)
    db.executescript('''CREATE TABLE aoi_burden AS SELECT source,aoi,SUM(burden) burden FROM behavior
      GROUP BY source,aoi HAVING SUM(burden)>0;
      CREATE UNIQUE INDEX aoi_burden_key ON aoi_burden(source,aoi);''')
    sql='''WITH totals AS (
      SELECT m.city,m.tier,m.cls,SUM(b.burden) total_burden,COUNT(*) aoi_count
      FROM aoi_burden b JOIN meta m USING(source,aoi) WHERE m.cls IN ('urban','rural')
      GROUP BY m.city,m.tier,m.cls),
      people AS (SELECT city,tier,cls,COUNT(DISTINCT userid) courier_count FROM courier_labels
      GROUP BY city,tier,cls),
      orders AS (SELECT city,tier,cls,SUM(n) order_count FROM matched GROUP BY city,tier,cls)
      SELECT t.*,p.courier_count,COALESCE(o.order_count,0) order_count FROM totals t
      LEFT JOIN people p USING(city,tier,cls) LEFT JOIN orders o USING(city,tier,cls)'''
    cities=pd.read_sql_query(sql,db)
    cities['burden_per_courier']=cities.total_burden/cities.courier_count.where(cities.courier_count>0)
    cities['burden_per_order']=cities.total_burden/cities.order_count.where(cities.order_count>0)
    cities['date']=date;cities.to_csv(work/f'city_classes_{date}.csv',index=False)
    maps=pd.read_sql_query('''WITH b AS (SELECT m.city,SUM(a.burden) total_burden FROM aoi_burden a
      JOIN meta m USING(source,aoi) GROUP BY m.city),
      o AS (SELECT city,SUM(n) order_count FROM matched GROUP BY city)
      SELECT b.*,COALESCE(o.order_count,0) order_count FROM b LEFT JOIN o USING(city)''',db)
    maps['burden_per_order']=maps.total_burden/maps.order_count.where(maps.order_count>0)
    maps.to_csv(work/f'city_map_{date}.csv',index=False)
    couriers=pd.read_sql_query('''WITH b AS (SELECT userid,SUM(burden) burden FROM behavior GROUP BY userid),
      ids AS (SELECT userid FROM b UNION SELECT userid FROM courier_orders),
      labels AS (SELECT DISTINCT userid,city,tier FROM courier_labels)
      SELECT i.userid,COALESCE(b.burden,0) burden,COALESCE(o.n,0) orders,r.city,r.tier
      FROM ids i LEFT JOIN b USING(userid) LEFT JOIN courier_orders o USING(userid)
      LEFT JOIN labels r ON i.userid=r.userid''',db)
    if couriers.tier.isna().any():raise ValueError('Missing supplied labels for behavior/order couriers')
    couriers.to_csv(work/f'couriers_{date}.csv',index=False)
    stats.update(couriers=len(couriers),tier_missing=int(couriers.tier.isna().sum()),city_classes=len(cities))
    stats['behavior_AOIs_missing_metadata']=db.execute('SELECT COUNT(*) FROM aoi_burden b LEFT JOIN meta m USING(source,aoi) WHERE m.aoi IS NULL').fetchone()[0]
    db.close();write_json(work/f'audit_{date}.json',stats)

# Pair periods
def pair_dates(work):
    a=pd.read_csv(work/'couriers_20240830.csv',dtype={'userid':str})
    b=pd.read_csv(work/'couriers_20241112.csv',dtype={'userid':str})
    d=a.merge(b[['userid','burden','orders']],on='userid',suffixes=('_regular','_promotion'),validate='one_to_one')
    valid=(d.orders_regular>0)&(d.burden_regular>0)
    stats={'common_couriers':len(d),'undefined_baseline':int((~valid).sum())}
    d=d.loc[valid].copy()
    d['order_change_pct']=100*(d.orders_promotion/d.orders_regular-1)
    d['burden_change_pct']=100*(d.burden_promotion/d.burden_regular-1)
    d.to_csv(work/'courier_changes_all_defined.csv',index=False)
    retained=(d.order_change_pct>-100)&(d.burden_change_pct>-100)
    stats['minus100_display_exclusions']=int((~retained).sum())
    d.loc[retained].to_csv(work/'courier_changes.csv',index=False)
    write_json(work/'pair_audit.json',stats)
