import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    events = pd.read_pickle(Path(interim) / 'events.pkl')

    hits = events.groupby(['GAME_ID', 'BAT_ID']).agg({
        'H_FL': 'max',
        'BAT_LINEUP_ID': 'first'
    })
    hits['Win'] = hits['H_FL'] > 0

    hits['Win_bin'] = hits['Win']*1

    hits = hits.reset_index()
    hits['Date'] = hits['GAME_ID'].str.slice(3,11)
    hits['Date'] = pd.to_datetime(hits['Date'], format='%Y%m%d')
    hits = hits.sort_values(['BAT_ID', 'Date', 'GAME_ID'])
    hits['cur_avg_win'] = hits.groupby('BAT_ID')['Win_bin'].transform(lambda x: x.rolling(200, 50).mean())
    hits['avg_win'] = hits.groupby('BAT_ID')['cur_avg_win'].shift(1)

    # hits['cur_exp_avg_win'] = hits.groupby('BAT_ID')['Win_bin'].ewm(com=0.5).mean()
    # hits['avg_exp_win'] = hits.groupby('BAT_ID')['cur_exp_avg_win'].shift(1)

    hits = hits.set_index(['GAME_ID', 'BAT_ID'])

    del hits['Date']

    hits.to_pickle(Path(interim) / 'hits.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
