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

    events['PIT_TEAM_ID'] = events['HOME_TEAM_ID']
    events.loc[events.BAT_HOME_ID == 1, 'PIT_TEAM_ID'] = events['AWAY_TEAM_ID']

    ptg = events.groupby(['PIT_TEAM_ID', 'GAME_ID']).agg({
        'H': 'sum',
        '2B': 'sum',
        '3B': 'sum',
        'HR': 'sum',
        'BB': 'sum',
        'IW': 'sum',
        'SO': 'sum',
        'HBP': 'sum',
        'EVENT_OUTS_CT': 'sum',
        'EVENT_RUNS_CT': 'sum',
        'Date': 'first',
    })

    ptg = ptg.rename(columns={
        'GAME_ID': 'G', 'AB_FL':'AB',
        'RBI_CT':'RBI', 'SH_FL':'SH', 'SF_FL':'SF'
    })

    ptg = ptg.rename(columns={'EVENT_OUTS_CT':'O', 'EVENT_RUNS_CT': 'R'})

    ptg['H']= ptg['H'].astype('Int8')
    ptg['2B']= ptg['2B'].astype('Int8')
    ptg['3B']= ptg['3B'].astype('Int8')
    ptg['HR']= ptg['HR'].astype('Int8')
    ptg['BB']= ptg['BB'].astype('Int8')
    ptg['IW']= ptg['IW'].astype('Int8')
    ptg['SO']= ptg['SO'].astype('Int8')
    ptg['HBP']= ptg['HBP'].astype('Int8')
    ptg['O']= ptg['O'].astype('Int8')
    ptg['R']= ptg['R'].astype('Int8')

    ptg['game_score'] = 47.4 + ptg['SO'] + 1.5*ptg['O'] - \
        2*ptg['BB'] - 2*ptg['H'] - 3*ptg['R'] - 4*ptg['HR']

    ptg = ptg.sort_values(['PIT_TEAM_ID', 'Date', 'GAME_ID'])
    ptg['cur_avg_game_score'] = ptg.groupby('PIT_TEAM_ID')['game_score'].transform(lambda x: x.rolling(20, 1).mean())
    ptg['avg_game_score'] = ptg.groupby('PIT_TEAM_ID')['cur_avg_game_score'].shift(1)
    ptg['avg_game_score'] = ptg['avg_game_score']

    del ptg['Date']

    ptg.to_pickle(Path(interim) / 'pitching_team_games.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
