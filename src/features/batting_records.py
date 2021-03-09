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

    events = events.merge(directory, left_on='BAT_ID',
                          right_on='PLAYER_ID', how='left')

    batting_record = events.groupby(['BAT_ID', 'year']).agg({
        'FirstName': 'first',
        'LastName': 'first',
        'BAT_HAND_CD': 'first',
        'GAME_ID': 'nunique',
        'AB_FL': lambda x: (x == 'T').sum(),
        'H_FL': [
            lambda x: (x > 0).sum(), # Hits
            lambda x: (x == 2).sum(), # Doubles
            lambda x: (x == 3).sum(), # Triples
            lambda x: (x == 4).sum() # Home Runs
            ],
        'RBI_CT': 'sum',
        'EVENT_CD': [
            lambda x: ((x == 14) | (x == 15)).sum(), # Walks
            lambda x: (x == 15).sum(), # Intentional Walks
            lambda x: (x == 3).sum(), # Strike Outs
            lambda x: (x == 16).sum() # Hit by Pitch
            ],
        'SH_FL': lambda x: (x == 'T').sum(),
        'SF_FL': lambda x: (x == 'T').sum(),
        })

    batting_record.columns = [
        'FirstName', 'LastName', 'BAT_HAND', 'G', 'AB', 'H', '2B', '3B', 'HR',
        'RBI', 'BB', 'IW', 'SO', 'HBP', 'SH', 'SF'
        ]



    hand = {'R': 1, 'L': 0}
    batting_record['BAT_HAND'] = batting_record['BAT_HAND'].map(hand)
    batting_record['BBPG'] = batting_record['BB'] / batting_record['G']
    batting_record['ABPG'] = batting_record['AB'] / batting_record['G']
    batting_record['HPG'] = batting_record['H'] / batting_record['G']
    batting_record.to_pickle(Path(interim) / 'batting_records.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
