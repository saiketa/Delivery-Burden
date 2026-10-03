# Figure sources
from pathlib import Path
import importlib.util
import sys
import sqlite3
import numpy as np
import pandas as pd

# Shared aggregation
def shared_prepare():
    folder=Path(__file__).resolve().parents[1]/'makefigure2'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('figure2_prepare',folder/'prepare.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

# Prepare spatial relationships
def prepare_sources(inputs,work):
    module=shared_prepare();module.prepare_date(inputs,'20240830',work)
    cities=pd.read_csv(work/'city_classes_20240830.csv')
    cities['burden_per_aoi']=cities.total_burden/cities.aoi_count.where(cities.aoi_count>0)
    cities.to_csv(work/'raincloud.csv',index=False)
    con=sqlite3.connect((work/'20240830.sqlite3').as_uri()+'?mode=ro',uri=True)
    burdens=pd.read_sql_query('SELECT source aoi_source_file,aoi aoiId,burden FROM aoi_burden',con)
    meta=pd.read_sql_query('SELECT source aoi_source_file,aoi aoiId,city,tier,cls FROM meta',con);con.close()
    population=pd.read_csv(inputs['aoi_csv'],usecols=['aoi_source_file','aoiId','population_persons'],dtype={'aoi_source_file':str,'aoiId':str})
    for col in ['aoi_source_file','aoiId']:population[col]=population[col].str.strip()
    population['population_persons']=pd.to_numeric(population.population_persons,errors='coerce')
    population=population.merge(meta,on=['aoi_source_file','aoiId'],how='left',validate='one_to_one')
    aoi=burdens.merge(population,on=['aoi_source_file','aoiId'],how='left',validate='one_to_one')
    aoi.to_csv(work/'aoi_population_burden.csv',index=False)
    census=pd.read_csv(inputs['city_population'],usecols=['city','population_persons','urban_population_persons'])
    census['city']=census.city.str.strip()
    for col in ['population_persons','urban_population_persons']:census[col]=pd.to_numeric(census[col],errors='coerce')
    if census.city.duplicated().any():raise ValueError('Repeated population city')
    urban=cities.loc[cities.cls.eq('urban')].merge(census,on='city',how='left',validate='one_to_one')
    urban['urban_population_10k']=urban.urban_population_persons/10000
    urban.to_csv(work/'urban_correlation.csv',index=False)
    city=pd.read_csv(work/'city_map_20240830.csv').merge(census,on='city',how='left',validate='one_to_one')
    tier=meta[['city','tier']].dropna(subset=['tier']).drop_duplicates()
    city=city.merge(tier,on='city',how='left',validate='one_to_one')
    city['population_10k']=city.population_persons/10000
    city.to_csv(work/'city_correlation.csv',index=False)
