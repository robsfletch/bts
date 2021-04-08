import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    game_logs = pd.read_pickle(Path(interim) / 'game_logs.pkl')

    # Select main columns for panel
    col_list = ['GAME_ID', 'date', 'year', 'ParkID', 'temp_park', '^home_team_id$',
                '^away_team_id$', '.*_lineup.*_bat_id', '.*_start_pit_id']
    p = '|'
    col_filter = p.join(col_list)
    game_logs = game_logs.filter(regex=col_filter)

    # Convert game logs list to panel
    panel = game_logs.melt(
        id_vars=['GAME_ID', 'date', 'year', 'ParkID', 'temp_park', 'away_start_pit_id',
            'home_start_pit_id', 'home_team_id', 'away_team_id'],
        var_name='spot', value_name='BAT_ID')

    ## Convert spot names into home dummy and lineup position
    panel['spot'] = panel['spot'].str.replace('_bat_id', '')
    panel[['home','spot']] = \
        panel['spot'].str.split(pat='_lineup', expand=True)
    panel['spot'] = panel['spot'].astype(int)

    dmap = {'home': 1, 'away': 0}
    panel['home'] = panel['home'].map(dmap).astype('Int8')

    ## Come up with corresponding IDs
    panel['PIT_ID'] = panel['home_start_pit_id']
    panel.loc[panel.home == True, 'PIT_ID'] = \
        panel['away_start_pit_id']

    panel['OWN_PIT_ID'] = panel['away_start_pit_id']
    panel.loc[panel.home == True, 'OWN_PIT_ID'] = \
        panel['home_start_pit_id']

    panel['PIT_TEAM_ID'] = panel['home_team_id']
    panel.loc[panel.home == True, 'PIT_TEAM_ID'] = \
        panel['away_team_id']

    panel['BAT_TEAM_ID'] = panel['away_team_id']
    panel.loc[panel.home == True, 'BAT_TEAM_ID'] = \
        panel['home_team_id']

    panel['spot'] = panel['spot'].astype('Int8')

    panel = panel[panel.year >= 1920]
    panel.to_pickle(Path(interim) / 'panel.pkl')
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
