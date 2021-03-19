import pandas as pd
from pathlib import Path
import click
import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    df = pd.read_csv(Path(input_filepath) / 'mlb_elo.csv')
    game_logs = pd.read_pickle(Path(output_filepath) / 'game_logs.pkl')

    df = df[df.season >= 1920]
    df['N'] = df.groupby(['date', 'team1', 'team2'])['date'].transform('count')
    df['n'] = df.groupby(['date', 'team1', 'team2'])['date'].cumcount() + 1
    df['group_id'] = df.groupby(['date', 'team1', 'team2']).ngroup()

    df = df.sort_values(['group_id', 'n'], ascending=[True, False])
    df['L_elo1_post'] = df.groupby(['group_id'])['elo1_post'].shift(1)
    df['equal'] = (df['L_elo1_post'] == df['elo1_pre']).astype('int')
    df['order'] = df.groupby(['group_id'])['equal'].cumsum() + 1
    df['double_id'] = 0
    df.loc[df.N >= 2 , 'double_id'] = df.order

    df['GAME_ID'] = df['team1'] + df['date'].str.replace('-', '') + df['double_id'].astype('str')

    game_list = game_logs.loc[game_logs.year >= 1920, ['GAME_ID', 'VisitingTeam', 'HomeTeam']]
    game_list.columns = ['GAME_ID_MAIN', 'AWAY_TEAM_ID', 'HOME_TEAM_ID']
    game_list['rest'] = game_list['GAME_ID_MAIN'].str.slice(3,12)
    game_list['NEW_HOME_TEAM_ID'] = game_list['HOME_TEAM_ID']
    game_list['NEW_AWAY_TEAM_ID'] = game_list['AWAY_TEAM_ID']

    recode = {
        'CHA': 'CHW',
        'NYA': 'NYY',
        'KCA': 'KCR',
        'NYN': 'NYM',
        'CHN': 'CHC',
        'LAN': 'LAD',
        'SLN': 'STL',
        'SFN': 'SFG',
        'SDN': 'SDP',
        'PHA': 'OAK',
        'BRO': 'LAD',
        'MON': 'WSN',
        'WS1': 'MIN',
        'NY1': 'SFG',
        'BSN': 'ATL',
        'CAL': 'ANA',
        'SLA': 'BAL',
        'TBA': 'TBD',
        'FLO': 'FLA',
        'MLN': 'ATL',
        'KC1': 'OAK',
        'WAS': 'WSN',
        'WS2': 'TEX',
        'MIA': 'FLA',
        'LAA': 'ANA',
        'SE1': 'MIL',
        }
    for old_code, new_code in recode.items():
        game_list.loc[game_list.NEW_HOME_TEAM_ID == old_code, 'NEW_HOME_TEAM_ID'] = new_code
        game_list.loc[game_list.NEW_AWAY_TEAM_ID == old_code, 'NEW_AWAY_TEAM_ID'] = new_code

    game_list['GAME_ID'] = game_list['NEW_HOME_TEAM_ID'] + game_list['rest']

    ratings538 = pd.merge(df, game_list, on=['GAME_ID'], indicator = True, how='right')

    ratings538 = ratings538[[
        'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2',
        'rating1_pre', 'rating2_pre', 'rating_prob1', 'rating_prob2',
        'pitcher1_rgs', 'pitcher2_rgs',  'pitcher1_adj', 'pitcher2_adj',
        'GAME_ID_MAIN', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'
    ]]

    ratings538.columns = [
        'elo_pre1', 'elo_pre2', 'elo_prob1', 'elo_prob2',
        'rating_pre1', 'rating_pre2', 'rating_prob1', 'rating_prob2',
        'pitcher_rgs1', 'pitcher_rgs2',  'pitcher_adj1', 'pitcher_adj2',
        'GAME_ID', 'TEAM_ID1', 'TEAM_ID2'
    ]

    ratings538_wide = pd.wide_to_long(
        ratings538,
        ['elo_pre', 'elo_prob', 'rating_pre', 'rating_prob', 'pitcher_rgs', 'pitcher_adj', 'TEAM_ID'],
        i='GAME_ID',
        j='Home'
    )

    ratings538_wide = ratings538_wide.reset_index()

    ratings538_wide['Home'] = 2 - ratings538_wide['Home']

    ratings538_wide = ratings538_wide.set_index(['GAME_ID', 'TEAM_ID'])

    ratings538_wide.to_pickle(Path(output_filepath) / 'ratings538.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
