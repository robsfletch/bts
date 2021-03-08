import pandas as pd
from pathlib import Path
import click
import logging

@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    clean_rosters = Path(interim) / 'rosters.pkl'
    rosters = pd.read_pickle(clean_rosters)

    directory = rosters[['PLAYER_ID', 'FirstName', 'LastName']]
    directory = directory.groupby(['PLAYER_ID']).agg('first')

    directory .to_pickle(Path(interim) / 'directory.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
