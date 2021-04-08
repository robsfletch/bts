import pandas as pd
from pathlib import Path
import click
import logging
import numpy as np


@click.command()
@click.argument('interim', type=click.Path(exists=True))
@click.argument('in_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
def main(interim, in_file, out_file):
    panel = pd.read_pickle(in_file)
    bg = pd.read_pickle(Path(interim) / 'batting_games.pkl')
    pg = pd.read_pickle(Path(interim) / 'pitching_games.pkl')
    ptg = pd.read_pickle(Path(interim) / 'pitching_team_games.pkl')
    br_predict = pd.read_pickle(Path(interim) / 'batting_records_predict.pkl')
    pr_predict = pd.read_pickle(Path(interim) / 'pitching_records_predict.pkl')
    btr_predict = pd.read_pickle(Path(interim) / 'batting_team_records_predict.pkl')
    ptr_predict = pd.read_pickle(Path(interim) / 'pitching_team_records_predict.pkl')
    ratings538 = pd.read_pickle(Path(interim) / 'ratings538.pkl')
    park_records = pd.read_pickle(Path(interim) / 'park_records.pkl')
    people = pd.read_pickle(Path(interim) / 'people.pkl')

    ## BATING HITS
    merged = panel.merge(
        bg[['Win', 'avg_win', 'PA_in_G']].add_prefix('b_'),
        on=['GAME_ID', 'BAT_ID'],
        how='left',
    )
    merged = merged.rename(columns={'b_Win':'Win'})

    ## PITCHING HITS
    merged = merged.merge(
        pg[['avg_game_score']].add_prefix('p_'),
        on=['GAME_ID', 'PIT_ID'],
        how='left',
    )

    merged = merged.merge(
        pg[['avg_game_score']].add_prefix('p_own_'),
        left_on=['GAME_ID', 'OWN_PIT_ID'],
        right_on=['GAME_ID', 'PIT_ID'],
        how='left',
    )

    merged = merged.merge(
        ptg[['avg_game_score']].add_prefix('p_team_'),
        left_on=['GAME_ID', 'PIT_TEAM_ID'],
        right_on=['GAME_ID', 'PIT_TEAM_ID'],
        how='left',
    )

    merged = merged.merge(
        ratings538[[
            'elo_pre', 'elo_prob', 'rating_pre', 'rating_prob',
            'pitcher_rgs']].add_prefix('rating_'),
        left_on=['GAME_ID', 'PIT_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )

    merged = merged.merge(
        ratings538[[
            'elo_pre', 'elo_prob', 'rating_pre', 'rating_prob',
            'pitcher_rgs']].add_prefix('rating_own_'),
        left_on=['GAME_ID', 'BAT_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
    )

    # Last season year
    merged['last_year'] = merged['year'] - 1

    ## BATTING SEASON RECORDS
    merged = merged.merge(
        br_predict, on=['BAT_ID', 'year'], how='left',
    )

    merged = merged.merge(
        people['bats'],
        left_on=['BAT_ID'],
        right_on = ['PlayerID'],
        how='left',
    )
    merged = merged.rename(columns={'bats':'BAT_HAND'})


    ## PITCHING SEASON RECORDS
    merged = merged.merge(
        pr_predict, on=['PIT_ID', 'year'], how='left',
    )

    merged = merged.merge(
        people['throws'],
        left_on=['PIT_ID'],
        right_on = ['PlayerID'],
        how='left',
    )
    merged = merged.rename(columns={'throws':'PIT_HAND'})


    merged = merged.merge(
        pr_predict.add_prefix('own_'),
        left_on = ['OWN_PIT_ID', 'year'],
        right_on=['PIT_ID', 'year'],
        how='left',
    )

    ## PITCHING TEAM SEASON RECORDS
    merged = merged.merge(
        ptr_predict[['p_team_pred_AdjHPG', 'p_team_pred_DefEff']],
        on=['PIT_TEAM_ID', 'year'],
        how='left'
    )

    ## BATTING TEAM SEASON RECORDS
    merged = merged.merge(
        btr_predict[['b_team_pred_AdjHPG']],
        on=['BAT_TEAM_ID', 'year'],
        how='left'
    )


    # PARK RECORDS
    merged = merged.merge(
        park_records[['h_factor']].add_prefix('park_'),
        left_on=['ParkID', 'last_year'],
        right_on=['ParkID', 'year'],
        how='left'
    )

    merged = merged.drop('last_year', 1)


    merged['b_pred_HPPA'] = merged['b_pred_AdjHPAdjPA'] * merged['park_h_factor']
    merged['p_pred_HPPA'] = merged['p_pred_AdjHPAdjPA'] * merged['park_h_factor']

    # merged['b_team_pred_AdjHPG'] = merged['b_team_pred_AdjHPG'] * merged['park_h_factor']
    # merged['p_team_pred_AdjHPG'] = merged['p_team_pred_AdjHPG'] * merged['park_h_factor']


    merged['opp_hands'] = 0
    merged['opp_hands'] = np.where(
        (merged['BAT_HAND'] == 1) & (merged['PIT_HAND'] == 0),
        1,
        merged['opp_hands']
    )

    merged['opp_hands'] = np.where(
        (merged['BAT_HAND'] == 0) & (merged['PIT_HAND'] == 1),
        1,
        merged['opp_hands']
    )

    merged.to_pickle(out_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
