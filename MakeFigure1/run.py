#!/usr/bin/env python3
# Figure entry point
import argparse
from pathlib import Path
from common import path

# Input paths
ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    '_base': ROOT,
    'behavior_files': [f'data/behavior_part_{i:02}.csv' for i in range(1,21)],
    'classified_orders_csv': 'data/orders_with_aoi.csv',
    'order_files': ['data/order_events_150_1.csv','data/order_events_150_2.csv','data/order_events_640.csv'],
    'minute_db': 'data/minute_aggregation.sqlite3',
    'aoi_csv': 'data/aois.csv',
    'city_boundaries': 'data/cities.geojson',
    'national_courier_points': 'data/courier_unique_points.csv',
    'national_point_datum': 'WGS84',
    'city_boundary_datum': 'WGS84',
    'district_boundary': 'data/hongkou.geojson',
    'district_datum': 'GCJ02',
    'aoi_datum': 'GCJ02',
    'district_courier_points': 'data/courier_unique_points_gcj02.csv',
    'district_courier_datum': 'GCJ02',
    'district_order_points': 'data/orders_with_aoi.csv',
    'district_order_datum': 'GCJ02',
    'work_directory': 'work',
    'output_directory': 'outputs',
}

# Run workflow
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare-burdens','prepare-a','prepare-minutes','plot'])
    a=parser.parse_args();c=INPUTS
    root=Path(__file__).resolve().parents[1]
    work=path(c,'work_directory').resolve();out=path(c,'output_directory').resolve()

    for p in [work,out]:
        if not p.is_relative_to(root):raise ValueError('work_directory and output_directory must be inside Delivery-Burden')
        p.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare-burdens':
        from prepare_burdens import run
        run(c,work)
    elif a.command=='prepare-a':
        from spatial import prepare_a
        prepare_a(c,work)
    elif a.command=='prepare-minutes':
        from prepare_minutes import run
        run(c,work)
    else:
        from plot import run
        run(c,work,out)

if __name__=='__main__':main()
