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

    hits.to_pickle(Path(interim) / 'hits.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
