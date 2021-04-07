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

    events['PIT_TEAM_ID'] = events['HOME_TEAM_ID']
    events.loc[events.BAT_HOME_ID == 1, 'PIT_TEAM_ID'] = events['AWAY_TEAM_ID']

    record = events.groupby(['PIT_TEAM_ID', 'year']).agg({
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
        'PA': 'sum',
        'AdjH': 'sum',
        'AdjPA': 'sum',
    })

    record = record.rename(columns={
        'GAME_ID': 'G', 'AB_FL':'AB',
        'RBI_CT':'RBI', 'SH_FL':'SH', 'SF_FL':'SF'
    })

    record['G']= record['G'].astype('Int16')
    record['AB']= record['AB'].astype('Int16')
    record['H']= record['H'].astype('Int16')
    record['2B']= record['2B'].astype('Int16')
    record['3B']= record['3B'].astype('Int16')
    record['HR']= record['HR'].astype('Int16')
    record['RBI']= record['RBI'].astype('Int16')
    record['BB']= record['BB'].astype('Int16')
    record['IW']= record['IW'].astype('Int8')
    record['SO']= record['SO'].astype('Int16')
    record['HBP']= record['HBP'].astype('Int8')
    record['SH']= record['SH'].astype('Int16')
    record['SF']= record['SF'].astype('Int16')

    record['AdjHPPA'] = record['AdjH'] / record['AdjPA']
    record['AdjHPG'] = record['AdjH'] / record['G']

    record.to_pickle(Path(interim) / 'pitching_team_records.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
