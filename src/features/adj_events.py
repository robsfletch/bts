import pandas as pd
from pathlib import Path
import click
import logging
import numpy as np


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    clean_events = Path(interim) / 'events.pkl'
    clean_park_records = (Path(interim) / 'park_records.pkl')
    clean_game_logs = (Path(interim) / 'game_logs.pkl')

    events = pd.read_pickle(clean_events)
    pr = pd.read_pickle(clean_park_records)
    gl = pd.read_pickle(clean_game_logs)

    events = pd.merge(
        events,
        gl[['GAME_ID', 'ParkID']],
        on=['GAME_ID'],
        how='left'
    )

    events = pd.merge(
        events,
        pr[['h_factor', 'pa_factor']],
        on=['ParkID', 'year'],
        how='left'
    )

    events['AdjH'] = events['H'] / events['h_factor']
    events['AdjPA'] = events['PA'] / events['pa_factor']

    events.to_pickle(Path(interim) / 'adj_events.pkl')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
