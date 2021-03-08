import pandas as pd
import os
import recordlinkage
from pathlib import Path
import numpy as np


class LineupProcessor():
    def __init__(self, date):
        self.__date = date
        self.__year = date[0:4]

    def scrape_data(self, input_filepath):
        spider = 'src/data/lineups/lineups/spiders/lineup_spider.py '
        os.system('python3 ' + spider + input_filepath + ' -d ' + self.__date)

    def read_data(self, input_filepath):
        """Read raw data into DataProcessor."""
        name = 'lineups' + self.__date + '.csv'
        lineups_csv = Path(input_filepath) / 'Lineups' / name
        df = pd.read_csv(lineups_csv)
        self.__raw_data = df

    def get_raw_data(self):
        return self.__raw_data

    def process_data(self, output_filepath, stable=True):
        """Process raw data into useful files for model."""

        clean_rosters = Path(output_filepath) / 'rosters.pkl'
        rosters = pd.read_pickle(clean_rosters)
        roster2019 = rosters[rosters.Year == self.__year]

        raw_lineup_wide = self.__raw_data

        raw_lineup_wide.drop(
            columns=['home_team_name', 'away_team_name'], inplace=True)

        raw_lineup_wide['id'] = np.arange(len(raw_lineup_wide))

        raw_lineup = raw_lineup_wide.melt(
            id_vars=['id', 'home_team_code', 'away_team_code'],
            var_name='lineup_id',
            value_name='Name')

        # Get First and Last Names
        raw_lineup['FirstName'] = raw_lineup.Name.str.split().str.get(0)
        raw_lineup['FirstName'] = \
            raw_lineup['FirstName'].str.replace('.', '', regex=False)
        raw_lineup['LastName'] = raw_lineup.Name.str.split().str.get(1)
        raw_lineup['LastName'] = \
            raw_lineup['LastName'].str.replace('.', '', regex=False)
        raw_lineup.drop(columns=['Name'], inplace=True)

        # set whether players are at home or away, and spot in lineup
        raw_lineup[['home', 'lineup_id']] = \
            raw_lineup['lineup_id'].str.split(pat='_', expand=True)

        # set team variable for each player
        raw_lineup['TEAM'] = raw_lineup['away_team_code']
        raw_lineup.loc[raw_lineup.home == 'home', 'TEAM'] = \
            raw_lineup['home_team_code']

        # set up record linking
        thresh = 0.4
        indexer = recordlinkage.Index()
        indexer.block(['TEAM', 'LastName'])
        pairs = indexer.index(raw_lineup, roster2019)

        # Generate matches
        compare = recordlinkage.Compare()
        compare.exact('TEAM', 'TEAM', label='Team')
        compare.exact('LastName', 'LastName', label='LastName')
        compare.string('FirstName', 'FirstName',
                       threshold=thresh, label='FirstName')
        features = compare.compute(pairs, raw_lineup, roster2019)

        # keep best matches
        lineup_with_id = features[features.sum(axis=1) >= 3]
        lineup_with_id = lineup_with_id.reset_index()[['level_0', 'level_1']]

        # merge matches
        lineup_with_id = lineup_with_id.join(
            roster2019, on='level_1', rsuffix='_roster')
        lineup_with_id = lineup_with_id.join(
            raw_lineup, on='level_0', rsuffix='_lineup')

        # clean up matched data
        lineup_with_id = lineup_with_id[[
            'PLAYER_ID', 'TEAM', 'lineup_id', 'home', 'id',
            'home_team_code', 'away_team_code'
        ]]
        lineup_with_id = lineup_with_id.set_index([
            'id', 'home_team_code', 'away_team_code', 'home', 'lineup_id'])
        lineup_with_id = lineup_with_id.reset_index()

        players_grid = lineup_with_id.pivot(
            index=['id', 'home_team_code', 'away_team_code'],
            columns=['lineup_id', 'home'],
            values=['PLAYER_ID'])
        players_grid.columns = players_grid.columns.droplevel()
        players_grid.columns = players_grid.columns.swaplevel(0, 1)
        players_grid.columns = [
            '_'.join(col).strip() for col in players_grid.columns.values
        ]
        players_grid = players_grid.set_index(
            ['home_pitcher', 'away_pitcher'], append=True
        )

        clean_lineups = players_grid.stack().reset_index().set_index([
            'home_team_code', 'away_team_code'
        ])
        clean_lineups.drop(columns=['id'], inplace=True)
        clean_lineups.columns = [
            'home_pitcher', 'away_pitcher', 'spot', 'BAT_ID'
        ]
        clean_lineups[['home', 'spot']] = \
            clean_lineups['spot'].str.split(pat='_', expand=True)
        clean_lineups['spot'] = clean_lineups['spot'].str.slice(start=-1)

        d = {'home': True, 'away': False}
        clean_lineups['home'] = clean_lineups['home'].map(d)
        clean_lineups

        self.__clean_data = clean_lineups

    def write_data(self, output_filepath):
        """Write processed data to directory."""
        save_file = Path(output_filepath) / 'lineup.pkl'
        self.__clean_data.to_pickle(save_file)
