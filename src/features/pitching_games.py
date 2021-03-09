import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    events = pd.read_pickle(Path(interim) / 'events.pkl')

    pg = events.groupby(['GAME_ID', 'PIT_ID']).agg({
        'H_FL': [
            lambda x: (x > 0).sum(), # Hits
            lambda x: (x == 2).sum(), # Doubles
            lambda x: (x == 3).sum(), # Triples
            lambda x: (x == 4).sum() # Home Runs
            ],
        'EVENT_CD': [
            lambda x: ((x == 14) | (x == 15)).sum(), # Walks
            lambda x: (x == 15).sum(), # Intentional Walks
            lambda x: (x == 3).sum(), # Strike Outs
            lambda x: (x == 16).sum() # Hit by Pitch
            ],
        'EVENT_OUTS_CT': 'sum',
        'EVENT_RUNS_CT': 'sum'
    })

    pg.columns = [
        'H', '2B', '3B', 'HR', 'BB', 'IW', 'SO', 'HBP', 'O', 'R'
        ]

    pg['game_score'] = 47.4 + pg['SO'] + 1.5*pg['O'] - \
        2*pg['BB'] - 2*pg['H'] - 3*pg['R'] - 4*pg['HR']

    pg = pg.reset_index()
    pg['Date'] = pg['GAME_ID'].str.slice(3,11)
    pg['Date'] = pd.to_datetime(pg['Date'], format='%Y%m%d')
    pg = pg.sort_values(['PIT_ID', 'Date', 'GAME_ID'])
    pg['cur_avg_game_score'] = pg.groupby('PIT_ID')['game_score'].transform(lambda x: x.rolling(20, 10).mean())
    pg['avg_game_score'] = pg.groupby('PIT_ID')['cur_avg_game_score'].shift(1)

    pg = pg.set_index(['GAME_ID', 'PIT_ID'])
    del pg['Date']

    pg.to_pickle(Path(interim) / 'pitching_games.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
