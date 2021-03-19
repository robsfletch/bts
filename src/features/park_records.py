import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    game_logs = pd.read_pickle(Path(interim) / 'game_logs.pkl')

    base = game_logs[['GAME_ID', 'HomeTeam', 'VisitingTeam', 'ParkID', 'HomeH', 'VisitorH']]
    base['year'] = base['GAME_ID'].str.slice(3, 7).astype('int')

    Home = base.groupby(['HomeTeam', 'year', 'ParkID']).agg({'HomeH':'sum', 'VisitorH':'sum', 'GAME_ID':'nunique'})
    Home.columns = ['home_hs', 'home_ha', 'home_g']
    Home['home_factor'] = (Home['home_hs'] + Home['home_ha']) / Home['home_g']
    Home = Home.rename_axis(['team', 'year', 'ParkID'])

    Road = base.groupby(['VisitingTeam', 'year']).agg({'HomeH':'sum', 'VisitorH':'sum', 'GAME_ID':'nunique'})
    Road.columns = ['road_ha', 'road_hs', 'road_g']
    Road['road_factor'] = (Road['road_hs'] + Road['road_ha']) / Road['road_g']
    Road = Road.rename_axis(['team', 'year'])

    park_factors = pd.merge(Home, Road, left_index=True, right_index=True)
    park_factors['factor_year'] = park_factors['home_factor'] / park_factors['road_factor']
    park_factors = park_factors.reset_index().set_index(['ParkID', 'year'])
    park_factors = park_factors.sort_values(['ParkID', 'year'])

    park_factors['factor'] = park_factors.groupby('ParkID')['factor_year'].transform(lambda x: x.rolling(10, 1).mean())
    park_factors['factor'] = park_factors['factor'].clip(.8, 1.25)

    park_factors.to_pickle(Path(interim) / 'park_records.pkl')
    print('finished')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
