import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    clean_events = Path(interim) / 'events.pkl'
    clean_directory = Path(interim) / 'directory.pkl'

    events = pd.read_pickle(clean_events)
    directory = pd.read_pickle(clean_directory)

    record = events.groupby(['PIT_ID', 'year']).agg({
        'FirstName': 'first',
        'LastName': 'first',
        'PIT_HAND_CD': 'first',
        'GAME_ID': 'nunique',
        'AB_FL': 'sum',
        'H_FL': [
            lambda x: (x > 0).sum(), # Hits
            lambda x: (x == 2).sum(), # Doubles
            lambda x: (x == 3).sum(), # Triples
            lambda x: (x == 4).sum() # Home Runs
            ],
        'EVENT_CD': [
            lambda x: ((x == 14) | (x == 15)).sum(), # Walks
            lambda x: (x == 15).sum(), # Intentional Walks
            lambda x: (x == 3).sum(), # Strike Outs
            lambda x: (x == 16).sum() # Hit by Pitch
            ],
        'SH_FL': 'sum',
        'SF_FL': 'sum',
    })

    record.columns = [
        'FirstName', 'LastName', 'PIT_HAND', 'G', 'AB', 'H', '2B', '3B',
        'HR', 'BB', 'IW', 'SO', 'HBP', 'SH', 'SF'
    ]

    record['G']= record['G'].astype('Int16')
    record['AB']= record['AB'].astype('Int8')
    record['H']= record['H'].astype('Int8')
    record['2B']= record['2B'].astype('Int8')
    record['3B']= record['3B'].astype('Int8')
    record['HR']= record['HR'].astype('Int8')
    record['BB']= record['BB'].astype('Int8')
    record['IW']= record['IW'].astype('Int8')
    record['SO']= record['SO'].astype('Int8')
    record['HBP']= record['HBP'].astype('Int8')
    record['SH']= record['SH'].astype('Int8')
    record['SF']= record['SF'].astype('Int8')

    hand = {'R': 1, 'L': 0}
    record['PIT_HAND'] = record['PIT_HAND'].map(hand)
    record['HPAB_p'] = record['H_p'] / record['AB_p']
    record.to_pickle(Path(interim) / 'pitching_records.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
