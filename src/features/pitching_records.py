import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    clean_adj_events = Path(interim) / 'adj_events.pkl'
    clean_directory = Path(interim) / 'directory.pkl'

    events = pd.read_pickle(clean_adj_events)
    directory = pd.read_pickle(clean_directory)

    events = events.merge(directory, left_on='PIT_ID',
                          right_on='PLAYER_ID', how='left')

    record = events.groupby(['PIT_ID', 'year']).agg({
        'FirstName': 'first',
        'LastName': 'first',
        'PIT_HAND_CD': 'first',
        'GAME_ID': 'nunique',
        'AB_FL': 'sum',
        'H': 'sum',
        '2B': 'sum',
        '3B': 'sum',
        'HR': 'sum',
        'BB': 'sum',
        'IW': 'sum',
        'SO': 'sum',
        'HBP': 'sum',
        'SH_FL': 'sum',
        'SF_FL': 'sum',
        'PA_NEW_FL': 'sum',
        'PA': 'sum',
        'AdjH': 'sum',
        'AdjPA': 'sum',
    })

    record = record.rename(columns={
        'PIT_HAND_CD':'PIT_HAND', 'GAME_ID': 'G', 'AB_FL':'AB',
        'SH_FL':'SH', 'SF_FL':'SF', 'PA_NEW_FL': 'PA_ALT'
    })

    record['G']= record['G'].astype('Int16')
    record['AB']= record['AB'].astype('Int16')
    record['H']= record['H'].astype('Int16')
    record['2B']= record['2B'].astype('Int8')
    record['3B']= record['3B'].astype('Int8')
    record['HR']= record['HR'].astype('Int8')
    record['BB']= record['BB'].astype('Int16')
    record['IW']= record['IW'].astype('Int8')
    record['SO']= record['SO'].astype('Int16')
    record['HBP']= record['HBP'].astype('Int8')
    record['SH']= record['SH'].astype('Int8')
    record['SF']= record['SF'].astype('Int8')
    record['PA']= record['PA'].astype('Int16')

    hand = {'R': 1, 'L': 0}
    record['PIT_HAND'] = record['PIT_HAND'].map(hand)

    record['HPPA'] = record['H'] / record['PA']
    record['HPAB'] = record['H'] / record['AB']
    record['HPAB'] = record['HPAB'].clip(.17, .35)
    record['AdjHPPA'] = record['AdjH'] / record['AdjPA']

    record.to_pickle(Path(interim) / 'pitching_records.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
