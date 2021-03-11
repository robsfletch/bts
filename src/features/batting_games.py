import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    events = pd.read_pickle(Path(interim) / 'events.pkl')

    bg = events.groupby(['GAME_ID', 'BAT_ID']).agg({
        'H': 'max',
        'BAT_LINEUP_ID': 'first',
        'Date': 'first',
    })

    bg = bg.rename(columns={'H': 'Win'})
    bg['Win'] = bg['Win'].astype('Int8')
    bg['BAT_LINEUP_ID'] = bg['BAT_LINEUP_ID'].astype('Int8')

    bg = bg.sort_values(['BAT_ID', 'Date', 'GAME_ID'])
    bg['cur_avg_win'] = bg.groupby('BAT_ID')['Win'].transform(lambda x: x.rolling(200, 50).mean())
    bg['avg_win'] = bg.groupby('BAT_ID')['cur_avg_win'].shift(1)

    # bg['cur_exp_avg_win'] = bg.groupby('BAT_ID')['Win_bin'].ewm(com=0.5).mean()
    # bg['avg_exp_win'] = bg.groupby('BAT_ID')['cur_exp_avg_win'].shift(1)

    del bg['Date']

    bg.to_pickle(Path(interim) / 'batting_games.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
