
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

    record = events.groupby(['BAT_ID', 'year']).agg({
        'FirstName': 'first',
        'LastName': 'first',
        'BAT_HAND_CD': 'first',
        'GAME_ID': 'nunique',
        'AB_FL': 'sum',
        'H': 'sum',
        '2B': 'sum',
        '3B': 'sum',
        'HR': 'sum',
        'RBI_CT': 'sum',
        'BB': 'sum',
        'IW': 'sum',
        'SO': 'sum',
        'HBP': 'sum',
        'SH_FL': 'sum',
        'SF_FL': 'sum',
        })

    record = record.rename(columns={
        'BAT_HAND_CD':'BAT_HAND', 'GAME_ID': 'G', 'AB_FL':'AB',
        'RBI_CT':'RBI', 'SH_FL':'SH', 'SF_FL':'SF'
    })

    record['G']= record['G'].astype('Int16')
    record['AB']= record['AB'].astype('Int16')
    record['H']= record['H'].astype('Int16')
    record['2B']= record['2B'].astype('Int8')
    record['3B']= record['3B'].astype('Int8')
    record['HR']= record['HR'].astype('Int8')
    record['RBI']= record['RBI'].astype('Int16')
    record['BB']= record['BB'].astype('Int16')
    record['IW']= record['IW'].astype('Int8')
    record['SO']= record['SO'].astype('Int16')
    record['HBP']= record['HBP'].astype('Int8')
    record['SH']= record['SH'].astype('Int8')
    record['SF']= record['SF'].astype('Int8')


    hand = {'R': 1, 'L': 0}
    record['BAT_HAND'] = record['BAT_HAND'].map(hand)
    record['BBPG'] = record['BB'] / record['G']
    record['ABPG'] = record['AB'] / record['G']
    record['HPG'] = record['H'] / record['G']

    record.to_pickle(Path(interim) / 'batting_records.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
