# Figure entry point
import argparse
from pathlib import Path
from prepare import prepare_date, pair_dates

# Input paths
ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    '20240830': {
        'aoi_csv': ROOT/'data/aois.csv',
        'courier_csv': ROOT/'data/couriers.csv',
        'behavior_files': [ROOT/f'data/behavior_part_{i:02}.csv' for i in range(1,21)],
        'orders_with_aoi': ROOT/'data/orders_with_aoi.csv',
        'order_files': [ROOT/f'data/order_events_{s}.csv' for s in ['150_1','150_2','640']],
    },
    '20241112': {
        'aoi_csv': ROOT/'data/20241112/aois.csv',
        'courier_csv': ROOT/'data/20241112/couriers.csv',
        'behavior_files': [ROOT/f'data/20241112/behavior_part_{i:02}.csv' for i in range(1,21)],
        'orders_with_aoi': ROOT/'data/20241112/orders_with_aoi.csv',
        'order_files': [ROOT/f'data/20241112/order_events_{s}.csv' for s in ['150_1','150_2','640']],
    },
}
BOUNDARIES = ROOT/'data/cities.geojson'
WORK = ROOT/'work/figure2'
OUTPUT = ROOT/'outputs/figure2'

# Run workflow
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','plot'])
    args=parser.parse_args()
    if args.command=='prepare':
        WORK.mkdir(parents=True,exist_ok=True)
        for date,inputs in PERIODS.items():prepare_date(inputs,date,WORK)
        pair_dates(WORK)
    else:
        from plot import make_figure
        OUTPUT.mkdir(parents=True,exist_ok=True)
        make_figure(WORK,BOUNDARIES,OUTPUT)

if __name__=='__main__':main()
