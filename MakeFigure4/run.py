# Figure entry point
import argparse
from pathlib import Path

# Input paths
ROOT=Path(__file__).resolve().parents[1]
INPUTS={
    'aoi_csv':ROOT/'data/aois.csv',
    'courier_csv':ROOT/'data/couriers.csv',
    'orders_with_aoi':ROOT/'data/orders_with_aoi.csv',
    'behavior_files':[ROOT/f'data/behavior_part_{i:02}.csv' for i in range(1,21)],
}
WORK=ROOT/'work/figure4'
OUTPUT=ROOT/'outputs/figure4'

# Run workflow
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','plot'])
    args=parser.parse_args()
    if args.command=='prepare':
        from prepare import prepare_sources
        WORK.mkdir(parents=True,exist_ok=True)
        prepare_sources(INPUTS,WORK)
    else:
        from plot import make_figure
        OUTPUT.mkdir(parents=True,exist_ok=True)
        make_figure(WORK,OUTPUT)

if __name__=='__main__':main()
