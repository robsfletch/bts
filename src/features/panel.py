import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    game_logs = pd.read_pickle(Path(interim) / 'game_logs.pkl')

    # Select main columns for panel
    col_list = ['GAME_ID', 'Date', 'year', 'ParkID', '^HomeTeam$',
                '^VisitingTeam$', '.*Batting.*PlayerID', '.*StartingPitcherID']
    p = '|'
    col_filter = p.join(col_list)
    game_logs = game_logs.filter(regex=col_filter)

    panel = game_logs.melt(
        id_vars=['GAME_ID', 'Date', 'year', 'ParkID', 'VisitorStartingPitcherID',
            'HomeStartingPitcherID', 'HomeTeam', 'VisitingTeam'],
        var_name='spot', value_name='BAT_ID')

    panel[['home','spot']] = \
        panel['spot'].str.split(pat='Batting', expand=True)
    panel['spot'] = panel['spot'].str.slice(stop=1).astype(int)

    d = {'Home': 1, 'Visitor': 0}
    panel['home'] = panel['home'].map(d).astype('Int8')

    panel['PIT_ID'] = panel['HomeStartingPitcherID']
    panel.loc[panel.home == True, 'PIT_ID'] = \
        panel['VisitorStartingPitcherID']

    panel['OWN_PIT_ID'] = panel['VisitorStartingPitcherID']
    panel.loc[panel.home == True, 'OWN_PIT_ID'] = \
        panel['HomeStartingPitcherID']

    panel['PIT_TEAM_ID'] = panel['HomeTeam']
    panel.loc[panel.home == True, 'PIT_TEAM_ID'] = \
        panel['VisitingTeam']

    panel['BAT_TEAM_ID'] = panel['VisitingTeam']
    panel.loc[panel.home == True, 'BAT_TEAM_ID'] = \
        panel['HomeTeam']

    panel['spot'] = panel['spot'].astype('Int8')

    panel = panel[panel.year >= 1920]
    panel.to_pickle(Path(interim) / 'panel.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
