import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    panel = pd.read_pickle(Path(interim) / 'panel.pkl')
    batting_games = pd.read_pickle(Path(interim) / 'batting_games.pkl')
    pitching_games = pd.read_pickle(Path(interim) / 'pitching_games.pkl')
    pitching_team_games = pd.read_pickle(Path(interim) / 'pitching_team_games.pkl')
    batting_records_predict = pd.read_pickle(Path(interim) / 'batting_records_predict.pkl')
    pitching_records = pd.read_pickle(Path(interim) / 'pitching_records.pkl')
    pitching_team_records = pd.read_pickle(Path(interim) / 'pitching_team_records.pkl')
    ratings538 = pd.read_pickle(Path(interim) / 'ratings538.pkl')
    park_records = pd.read_pickle(Path(interim) / 'park_records.pkl')

    ## BATING HITS
    merged = panel.merge(
        batting_games[['Win', 'avg_win']].add_prefix('b_'),
        on=['GAME_ID', 'BAT_ID'],
        how='left',
    )
    merged = merged.rename(columns={'b_Win':'Win'})

    ## PITCHING HITS
    merged = merged.merge(
        pitching_games[['avg_game_score']].add_prefix('p_'),
        on=['GAME_ID', 'PIT_ID'],
        how='left',
    )

    merged = merged.merge(
        pitching_games[['avg_game_score']].add_prefix('p_own_'),
        left_on=['GAME_ID', 'OWN_PIT_ID'],
        right_on=['GAME_ID', 'PIT_ID'],
        how='left',
    )

    merged = merged.merge(
        pitching_team_games[['avg_game_score']].add_prefix('p_team_'),
        left_on=['GAME_ID', 'PIT_TEAM_ID'],
        right_on=['GAME_ID', 'PIT_TEAM_ID'],
        how='left',
    )

    merged = merged.merge(
        ratings538[[
            'elo_pre', 'elo_prob', 'rating_pre', 'rating_prob',
            'pitcher_rgs', 'pitcher_adj']].add_prefix('rating_'),
        left_on=['GAME_ID', 'PIT_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
    )

    merged = merged.merge(
        ratings538[[
            'elo_pre', 'elo_prob', 'rating_pre', 'rating_prob',
            'pitcher_rgs', 'pitcher_adj']].add_prefix('rating_own_'),
        left_on=['GAME_ID', 'BAT_TEAM_ID'],
        right_on=['GAME_ID', 'TEAM_ID'],
        how='left',
    )

    # Last season year
    merged['last_year'] = merged['year'] - 1

    ## BATTING SEASON RECORDS
    merged = merged.merge(
        batting_records_predict, on=['BAT_ID', 'year'], how='left',
    )

    ## PITCHING SEASON RECORDS
    merged = merged.merge(
        pitching_records[['PIT_HAND', 'G', 'HPPA', 'HPAB']].add_prefix('p_'),
        left_on=['PIT_ID', 'last_year'],
        right_on=['PIT_ID', 'year'],
        how='left'
    )
    merged = merged.rename(columns={'p_PIT_HAND':'PIT_HAND'})

    merged = merged.merge(
        pitching_records[['G', 'HPPA', 'HPAB']].add_prefix('p_own_'),
        left_on = ['OWN_PIT_ID', 'last_year'],
        right_on=['PIT_ID', 'year'],
        how='left',
    )

    ## PITCHING TEAM SEASON RECORDS
    merged = merged.merge(
        pitching_team_records[['HPG', 'BBPG', 'ABPG', 'HPAB', 'HPPA']].add_prefix('p_team_'),
        left_on=['PIT_TEAM_ID', 'last_year'],
        right_on=['PIT_TEAM_ID', 'year'],
        how='left'
    )

    ## PARK RECORDS
    merged = merged.merge(
        park_records[['factor']].add_prefix('park_'),
        left_on=['ParkID', 'last_year'],
        right_on=['ParkID', 'year'],
        how='left'
    )

    merged = merged.drop('last_year', 1)
    merged.to_pickle(Path(interim) / 'merged_data.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
