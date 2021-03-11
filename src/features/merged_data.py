import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    panel = pd.read_pickle(Path(interim) / 'panel.pkl')
    hits = pd.read_pickle(Path(interim) / 'batting_games.pkl')
    batting_records = pd.read_pickle(Path(interim) / 'batting_records.pkl')
    pitching_records = pd.read_pickle(Path(interim) / 'pitching_records.pkl')
    park_records = pd.read_pickle(Path(interim) / 'park_records.pkl')

    merged_data = panel.merge(hits, on=['GAME_ID', 'BAT_ID'])
    merged_data['last_year'] = merged_data['year'] - 1
    batting_records = batting_records.reset_index()
    batting_records = batting_records.rename(columns={'year':'last_year'})
    merged_data = merged_data.merge(
        batting_records, on=['BAT_ID', 'last_year'], how='left')

    pitching_records = pitching_records.reset_index()
    pitching_records = pitching_records.rename(columns={'year':'last_year',})
    merged_data = merged_data.merge(
        pitching_records, on=['PIT_ID', 'last_year'], how='left')

    park_records = park_records.reset_index()
    park_records = park_records.rename(columns={'year':'last_year',})
    merged_data = merged_data.merge(
        park_records, on=['ParkID', 'last_year'], how='left')

    merged_data = merged_data.drop('last_year', 1)
    merged_data.to_pickle(Path(interim) / 'merged_data.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
