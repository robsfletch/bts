import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
import statsmodels.formula.api as sm
import patsy


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    game_logs = pd.read_pickle(Path(interim) / 'game_logs.pkl')

    park_factors = fe(game_logs)

    park_factors.to_pickle(Path(interim) / 'park_records.pkl')
    print('finished')

def espn(game_logs):
    base = game_logs.loc[: ,['GAME_ID', 'HomeTeam', 'VisitingTeam', 'ParkID', 'HomeH', 'VisitorH']]
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

    park_factors['factor'] = park_factors.groupby('ParkID')['factor_year'].transform(lambda x: x.rolling(5, 1).mean())
    park_factors['factor'] = park_factors['factor'].clip(.8, 1.25)

    return park_factors

def fe(gl):
    gl = gl.loc[:, ['GAME_ID', 'HomeH', 'VisitorH', 'ParkID', 'HomeTeam', 'VisitingTeam', 'year']]
    gl = gl.astype({
        'HomeH': 'float',
        'VisitorH': 'float',
        'ParkID': 'object',
        'HomeTeam':'object',
        'VisitingTeam': 'object',
    })
    gl = gl.rename(columns={
        'HomeH': 'H_Home',
        'VisitorH': 'H_Visitor',
    })
    gl = gl.dropna()

    gl = pd.wide_to_long(
        gl , ['H'] , i=['GAME_ID', 'year', 'HomeTeam', 'VisitingTeam'],
        j='Home', sep='_', suffix=r'\w+'
    )

    gl = gl.reset_index().set_index(['GAME_ID'])
    gl['BatTeam'] = np.where(
        gl['Home'] == 'Home',
        gl['HomeTeam'],
        gl['VisitingTeam']
    )
    gl['FieldTeam'] = np.where(
        gl['Home'] == 'Home',
        gl['VisitingTeam'],
        gl['HomeTeam']
    )

    gl = gl[['year', 'ParkID', 'H', 'BatTeam', 'FieldTeam']]

    gl['Log_H'] = np.log(gl['H'] + 1)

    park_factors = pd.DataFrame(columns=['ParkID', 'factor_year', 'year'])

    max_year = gl.year.max() + 1
    min_year = gl.year.min()
    for year in range(1920, max_year):
        gl_year = gl.loc[gl.year == year]

        result = sm.ols(formula="Log_H ~ C(ParkID) + C(BatTeam) + C(FieldTeam)", data=gl_year).fit()

        test = pd.DataFrame(result.params)
        test = test.reset_index()
        test.columns = ['ParkID', 'factor_year']

        ParkFactors = test.loc[test.ParkID.str.contains('ParkID'), ['ParkID', 'factor_year']]

        ParkFactors['ParkID'] = ParkFactors.ParkID.str.slice(start=-6, stop=-1)

        mat = patsy.dmatrix("C(ParkID)", gl_year)
        di = mat.design_info
        fi = di.factor_infos
        fi2 = fi[list(di.factor_infos.keys())[0]]
        omitted = fi2.categories[0]

        ParkFactors = ParkFactors.append({'ParkID':omitted, 'factor_year':0}, ignore_index=True)
        ParkFactors['year'] = year

        park_factors = park_factors.append(ParkFactors)

    park_factors = park_factors.sort_values(['ParkID', 'year'])
    park_factors = park_factors.set_index(['ParkID', 'year'])

    park_factors['factor_year'] = park_factors['factor_year'] + 1

    park_factors['factor'] = park_factors.groupby('ParkID')['factor_year'].transform(lambda x: x.rolling(5, 1).mean())

    return park_factors

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
