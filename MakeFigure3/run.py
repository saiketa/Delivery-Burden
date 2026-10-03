# Figure entry point
import argparse
from pathlib import Path

# Input paths
ROOT=Path(__file__).resolve().parents[1]
INPUTS={
    'aoi_csv':ROOT/'data/aois.csv',
    'courier_csv':ROOT/'data/couriers.csv',
    'behavior_files':[ROOT/f'data/behavior_part_{i:02}.csv' for i in range(1,21)],
    'orders_with_aoi':ROOT/'data/orders_with_aoi.csv',
    'order_files':[ROOT/f'data/order_events_{s}.csv' for s in ['150_1','150_2','640']],
    'city_population':ROOT/'data/city_population.csv',
}
WORK=ROOT/'work/figure3'
OUTPUT=ROOT/'outputs/figure3'

# Run workflow
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','plot'])
    args=parser.parse_args()
    if args.command=='prepare':
        from prepare import prepare_sources
        WORK.mkdir(parents=True,exist_ok=True);prepare_sources(INPUTS,WORK)
    else:
        from plot import make_figure
        OUTPUT.mkdir(parents=True,exist_ok=True);make_figure(WORK,OUTPUT)

if __name__=='__main__':main()
