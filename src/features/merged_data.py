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
    batting_records = pd.read_pickle(Path(interim) / 'batting_records.pkl')
    pitching_records = pd.read_pickle(Path(interim) / 'pitching_records.pkl')
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
        pitching_games[['avg_game_score']].add_prefix('p_team_'),
        left_on=['GAME_ID', 'TEAM_PIT_ID'],
        right_on=['GAME_ID', 'PIT_ID'],
        how='left',
    )

    # Last season year
    merged['last_year'] = merged['year'] - 1

    ## BATTING SEASON RECORDS
    merged = merged.merge(
        batting_records[['BAT_HAND', 'G', 'BBPG', 'ABPG', 'HPG']].add_prefix('b_'),
        left_on=['BAT_ID', 'last_year'],
        right_on=['BAT_ID', 'year'],
        how='left',
    )
    merged = merged.rename(columns={'b_BAT_HAND':'BAT_HAND'})

    ## PITCHING SEASON RECORDS
    merged = merged.merge(
        pitching_records[['PIT_HAND', 'G', 'HPPA']].add_prefix('p_'),
        left_on=['PIT_ID', 'last_year'],
        right_on=['PIT_ID', 'year'],
        how='left'
    )
    merged = merged.rename(columns={'p_PIT_HAND':'PIT_HAND'})

    merged = merged.merge(
        pitching_records[['G', 'HPPA']].add_prefix('p_team_'),
        left_on = ['TEAM_PIT_ID', 'last_year'],
        right_on=['PIT_ID', 'year'],
        how='left',
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
