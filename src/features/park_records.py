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

    park_factors = espn(game_logs)

    park_factors.to_pickle(Path(interim) / 'park_records.pkl')
    print('finished')

def espn(game_logs):
    base = game_logs.loc[: ,[
        'GAME_ID', 'ParkID', 'home_team_id', 'away_team_id',
        'home_h', 'away_h', 'home_pa', 'away_pa'
    ]]
    base['year'] = base['GAME_ID'].str.slice(3, 7).astype('int')
    base = base.loc[base.year >= 1920]

    import pdb; pdb.set_trace()
    ## Calculate Home factors
    Home = base.groupby(['home_team_id', 'year', 'ParkID']).agg({
        'home_h':'sum',
        'away_h':'sum',
        'home_pa':'sum',
        'away_pa':'sum',
        'GAME_ID':'nunique'
    })
    Home = Home.rename(columns = {
        'home_h': 'home_h_s',
        'away_h': 'home_h_a',
        'home_pa': 'home_pa_s',
        'away_pa': 'home_pa_a',
        'GAME_ID': 'home_g',
    })
    Home['home_h_factor'] = (
        (Home['home_h_s'] + Home['home_h_a']) / Home['home_g']
    )
    Home['home_pa_factor'] = (
        (Home['home_pa_s'] + Home['home_pa_a']) / Home['home_g']
    )
    Home = Home.rename_axis(['team', 'year', 'ParkID'])

    ## Calculate road factors
    Road = base.groupby(['away_team_id', 'year']).agg({
        'home_h':'sum',
        'away_h':'sum',
        'home_pa':'sum',
        'away_pa':'sum',
        'GAME_ID':'nunique'
    })
    Road = Road.rename(columns = {
        'home_h': 'road_h_a',
        'away_h': 'road_h_s',
        'home_pa': 'road_pa_a',
        'away_pa': 'road_pa_s',
        'GAME_ID': 'road_g',
    })
    Road['road_h_factor'] = (
        (Road['road_h_s'] + Road['road_h_a']) / Road['road_g']
    )
    Road['road_pa_factor'] = (
        (Road['road_pa_s'] + Road['road_pa_a']) / Road['road_g']
    )
    Road = Road.rename_axis(['team', 'year'])


    ## Calculate park factors from home and road factors
    park_factors = pd.merge(Home, Road, left_index=True, right_index=True)
    park_factors['factor_h_year'] = park_factors['home_h_factor'] / park_factors['road_h_factor']
    park_factors['factor_pa_year'] = park_factors['home_pa_factor'] / park_factors['road_pa_factor']
    park_factors['factor_hppa_year'] = park_factors['factor_h_year'] / park_factors['factor_pa_year']
    park_factors = park_factors.reset_index().set_index(['ParkID', 'year'])
    park_factors = park_factors.sort_values(['ParkID', 'year'])

    park_factors['h_factor'] = park_factors.groupby('ParkID')[['factor_h_year']].transform(lambda x: x.rolling(10, 1).mean())
    park_factors['pa_factor'] = park_factors.groupby('ParkID')[['factor_pa_year']].transform(lambda x: x.rolling(10, 1).mean())
    park_factors['hppa_factor'] = park_factors.groupby('ParkID')[['factor_hppa_year']].transform(lambda x: x.rolling(10, 1).mean())

    park_factors['h_factor'] = park_factors['h_factor'].clip(.8, 1.25)

    idx = park_factors.groupby(['ParkID', 'year'])['home_g'].transform(max) == park_factors['home_g']
    park_factors = park_factors.loc[idx]
    park_factors = park_factors.loc[park_factors.home_g >= 20]

    return park_factors
#
# def fe(gl):
#     gl = gl.loc[:, ['GAME_ID', 'home_h', 'away_h', 'ParkID', 'home_team_id', 'away_team_id', 'year']]
#     gl = gl.astype({
#         'home_h': 'float',
#         'away_h': 'float',
#         'ParkID': 'object',
#         'home_team_id':'object',
#         'away_team_id': 'object',
#     })
#     gl = gl.rename(columns={
#         'home_h': 'H_Home',
#         'away_h': 'H_Visitor',
#     })
#     gl = gl.dropna()
#
#     gl = pd.wide_to_long(
#         gl , ['H'] , i=['GAME_ID', 'year', 'home_team_id', 'away_team_id'],
#         j='Home', sep='_', suffix=r'\w+'
#     )
#
#     gl = gl.reset_index().set_index(['GAME_ID'])
#     gl['BatTeam'] = np.where(
#         gl['Home'] == 'Home',
#         gl['home_team_id'],
#         gl['away_team_id']
#     )
#     gl['FieldTeam'] = np.where(
#         gl['Home'] == 'Home',
#         gl['away_team_id'],
#         gl['home_team_id']
#     )
#
#     gl = gl[['year', 'ParkID', 'H', 'BatTeam', 'FieldTeam']]
#
#     gl['Log_H'] = np.log(gl['H'] + 1)
#
#     park_factors = pd.DataFrame(columns=['ParkID', 'factor_year', 'year'])
#
#     max_year = gl.year.max() + 1
#     min_year = gl.year.min()
#     for year in range(1920, max_year):
#         gl_year = gl.loc[gl.year == year]
#
#         result = sm.ols(formula="Log_H ~ C(ParkID) + C(BatTeam) + C(FieldTeam)", data=gl_year).fit()
#
#         test = pd.DataFrame(result.params)
#         test = test.reset_index()
#         test.columns = ['ParkID', 'factor_year']
#
#         ParkFactors = test.loc[test.ParkID.str.contains('ParkID'), ['ParkID', 'factor_year']]
#
#         ParkFactors['ParkID'] = ParkFactors.ParkID.str.slice(start=-6, stop=-1)
#
#         mat = patsy.dmatrix("C(ParkID)", gl_year)
#         di = mat.design_info
#         fi = di.factor_infos
#         fi2 = fi[list(di.factor_infos.keys())[0]]
#         omitted = fi2.categories[0]
#
#         ParkFactors = ParkFactors.append({'ParkID':omitted, 'factor_year':0}, ignore_index=True)
#         ParkFactors['year'] = year
#
#         park_factors = park_factors.append(ParkFactors)
#
#     park_factors = park_factors.sort_values(['ParkID', 'year'])
#     park_factors = park_factors.set_index(['ParkID', 'year'])
#
#     park_factors['factor_year'] = park_factors['factor_year'] + 1
#
#     park_factors['factor'] = park_factors.groupby('ParkID')['factor_year'].transform(lambda x: x.rolling(5, 1).mean())
#
#     return park_factors

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
