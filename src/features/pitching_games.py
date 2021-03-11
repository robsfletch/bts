import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    events = pd.read_pickle(Path(interim) / 'events.pkl')

    pg = events.groupby(['GAME_ID', 'PIT_ID']).agg({
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

    pg = pg.rename(columns={'EVENT_OUTS_CT':'O', 'EVENT_RUNS_CT': 'R'})

    pg['H']= pg['H'].astype('Int8')
    pg['2B']= pg['2B'].astype('Int8')
    pg['3B']= pg['3B'].astype('Int8')
    pg['HR']= pg['HR'].astype('Int8')
    pg['BB']= pg['BB'].astype('Int8')
    pg['IW']= pg['IW'].astype('Int8')
    pg['SO']= pg['SO'].astype('Int8')
    pg['HBP']= pg['HBP'].astype('Int8')
    pg['O']= pg['O'].astype('Int8')
    pg['R']= pg['R'].astype('Int8')

    pg['game_score'] = 47.4 + pg['SO'] + 1.5*pg['O'] - \
        2*pg['BB'] - 2*pg['H'] - 3*pg['R'] - 4*pg['HR']

    pg = pg.sort_values(['PIT_ID', 'Date', 'GAME_ID'])
    pg['cur_avg_game_score'] = pg.groupby('PIT_ID')['game_score'].transform(lambda x: x.rolling(20, 1).mean())
    pg['avg_game_score'] = pg.groupby('PIT_ID')['cur_avg_game_score'].shift(1)

    del pg['Date']

    pg.to_pickle(Path(interim) / 'pitching_games.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
