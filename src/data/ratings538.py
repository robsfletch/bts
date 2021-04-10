import pandas as pd
from pathlib import Path
import click
import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    df = pd.read_csv(Path(input_filepath) / 'mlb_elo.csv')
    teams = pd.read_pickle(Path(output_filepath) / 'teams.pkl')

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

    df = df.rename(columns={'season':'year'})

    ##### MAP team codes to retrosheets team codes
    team_map = gen_team_id_map(df, teams)

    ratings538 = pd.merge(
        df.reset_index(),
        team_map[['TEAM_ID', 'year', 'teamIDretro']],
        left_on=['team1', 'year'],
        right_on=['TEAM_ID', 'year'],
        how='left',
    )

    del ratings538['TEAM_ID']
    del ratings538['team1']
    ratings538 = ratings538.rename(columns={'teamIDretro': 'team1'})

    ratings538 = pd.merge(
        ratings538.reset_index(),
        team_map[['TEAM_ID', 'year', 'teamIDretro']],
        left_on=['team2', 'year'],
        right_on=['TEAM_ID', 'year'],
        how='left',
    )

    del ratings538['TEAM_ID']
    del ratings538['team2']
    ratings538 = ratings538.rename(columns={'teamIDretro': 'team2'})

    ### Generate GAME ID
    ratings538['GAME_ID'] = ratings538['team1'] + \
        ratings538['date'].str.replace('-', '') + \
        ratings538['double_id'].astype('str')

    ratings538 = ratings538.drop_duplicates(subset=['GAME_ID'])
    ratings538 = ratings538[[
        'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2',
        'rating1_pre', 'rating2_pre', 'rating_prob1', 'rating_prob2',
        'pitcher1_rgs', 'pitcher2_rgs',  'pitcher1_adj', 'pitcher2_adj',
        'GAME_ID', 'team1', 'team2', 'year'
    ]]

    ratings538.columns = [
        'elo_pre1', 'elo_pre2', 'elo_prob1', 'elo_prob2',
        'rating_pre1', 'rating_pre2', 'rating_prob1', 'rating_prob2',
        'pitcher_rgs1', 'pitcher_rgs2',  'pitcher_adj1', 'pitcher_adj2',
        'GAME_ID', 'TEAM_ID1', 'TEAM_ID2', 'year'
    ]

    ratings538_wide = pd.wide_to_long(
        ratings538,
        ['elo_pre', 'elo_prob', 'rating_pre', 'rating_prob', 'pitcher_rgs', 'pitcher_adj', 'TEAM_ID'],
        i=['GAME_ID', 'year'],
        j='Home'
    )

    ratings538_wide = ratings538_wide.reset_index()

    ratings538_wide['Home'] = 2 - ratings538_wide['Home']

    ratings538_wide = ratings538_wide.set_index(['GAME_ID', 'TEAM_ID'])

    ratings538_wide.to_pickle(Path(output_filepath) / 'ratings538.pkl')

def gen_team_id_map(ratings538, teams):
    ratings_teams = ratings538.groupby(['team1', 'year']).first()
    ratings_teams = ratings_teams.reset_index()[['team1', 'year']]
    ratings_teams.columns = ['TEAM_ID', 'year']

    recode = {
        'MIA': 'FLA',
        'NYG': 'SFG',
        'BRO': 'LAD',
        'MON': 'WSN',
        'SLB': 'BAL',
        'TBR': 'TBD',
        'WSA': 'TEX',
        'SEP': 'MIL',
        'CAL': 'ANA',
        'LAA': 'ANA',
        'WSH': 'MIN',
        'KCA': 'OAK',
        'PHA': 'OAK',
        'BSN': 'ATL',
        'MLN': 'ATL',
    }
    for old_code, new_code in recode.items():
        teams.loc[teams.teamIDBR == old_code, 'teamIDBR'] = new_code

    teams2021 = teams.loc[teams.year == 2020, :]
    teams2021.loc[:, ['year']] = 2021
    teams = teams.append(teams2021)

    team_map = pd.merge(
        ratings_teams,
        teams.loc[:, ['year', 'teamIDretro', 'teamIDBR']],
        left_on = ['year', 'TEAM_ID'],
        right_on = ['year', 'teamIDBR'],
    )

    return team_map


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
