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

    events = events.merge(directory, left_on='PIT_ID',
                          right_on='PLAYER_ID', how='left')

    pitching_record = events.groupby(['PIT_ID', 'year']).agg({
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

    pitching_record.columns = [
        'FirstName_p', 'LastName_p', 'PIT_HAND', 'G_p', 'AB_p', 'H_p', '2B_p', '3B_p', 'HR_p',
        'BB_p', 'IW_p', 'SO_p', 'HBP_p', 'SH_p', 'SF_p'
        ]

    hand = {'R': 1, 'L': 0}
    pitching_record['PIT_HAND'] = pitching_record['PIT_HAND'].map(hand)
    pitching_record['HPAB_p'] = pitching_record['H_p'] / pitching_record['AB_p']
    pitching_record.to_pickle(Path(interim) / 'pitching_records.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
