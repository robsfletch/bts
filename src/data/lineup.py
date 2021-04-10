import pandas as pd
import os
import recordlinkage
from pathlib import Path
import numpy as np


class LineupProcessor():
    def __init__(self, date):
        self.__date = date
        self.__year = date[0:4]

    def full_prep(self, input_filepath, output_filepath, tomorrow='True'):
        self.scrape_rotowire(input_filepath, tomorrow)
        self.read_data(input_filepath)
        self.process_data(output_filepath)
        self.write_data(output_filepath)

    def scrape_data(self, input_filepath):
        spider = 'src/data/lineups/lineups/spiders/lineup_spider.py '
        os.system('python3 ' + spider + input_filepath + ' -d ' + self.__date)

    def scrape_rotowire(self, input_filepath, tomorrow):
        spider = 'src/data/lineups/lineups/spiders/rotowire_lineup_spider.py '
        os.system('python3 ' + spider + input_filepath + ' -d ' + self.__date + ' -t' + tomorrow)

    def read_data(self, input_filepath):
        """Read raw data into DataProcessor."""
        name = 'lineups' + self.__date + '.csv'
        lineups_csv = Path(input_filepath) / 'Lineups' / name
        df = pd.read_csv(lineups_csv)
        self.__raw_data = df

    def get_raw_data(self):
        return self.__raw_data

    def process_data(self, output_filepath, stable=True):
        clean_rosters = Path(output_filepath) / 'rosters.pkl'
        rosters = pd.read_pickle(clean_rosters)
        rosters['base'] = rosters['PLAYER_ID'].str[0:6]
        rosters = rosters.loc[rosters.year >= 2015]
        rosters = rosters.dropna()
        rosters = rosters.sort_values(['PLAYER_ID', 'year'], ascending = [True, False])
        roster_match = rosters.groupby('PLAYER_ID')[['base', 'FirstName', 'LastName', 'TEAM']].first().reset_index()
        roster_match.columns = ['PLAYER_ID', 'base', 'FirstName', 'LastName', 'TEAM_2020']

        park_records = pd.read_pickle(Path(output_filepath) / 'park_records.pkl')
        pr2020 = park_records.loc[
            (park_records.index.get_level_values('year') == 2020), ['team']
        ]
        pr2020 = pr2020.reset_index()
        pr2020 = pr2020.loc[:, ['ParkID', 'team']]
        pr2020 = pr2020.set_index('team')

        raw_lineup_wide = self.__raw_data
        raw_lineup_wide = raw_lineup_wide.drop_duplicates()
        recode = {
            'STL': 'SLN',
            'LAD': 'LAN',
            'TB': 'TBA',
            'NYM': 'NYN',
            'CWS': 'CHA',
            'SF': 'SFN',
            'NYY': 'NYA',
            'KC': 'KCA',
            'SD': 'SDN',
            'LAA': 'ANA',
            'CHC': 'CHN',
        }
        for old_code, new_code in recode.items():
            raw_lineup_wide.loc[
                raw_lineup_wide.home_team_code == old_code,
                'home_team_code'
            ] = new_code

            raw_lineup_wide.loc[
                raw_lineup_wide.away_team_code == old_code,
                'away_team_code'
            ] = new_code

        raw_lineup_wide['game_time'] = pd.to_datetime(raw_lineup_wide.game_time)
        raw_lineup_wide = raw_lineup_wide.sort_values(['home_team_code', 'game_time'])
        raw_lineup_wide['NumGames'] = raw_lineup_wide.groupby(['home_team_code'])['game_time'].transform('count')
        raw_lineup_wide['GameOrder'] = raw_lineup_wide.groupby(['home_team_code']).cumcount() + 1

        raw_lineup_wide['DoubleHeader'] = np.where(
            raw_lineup_wide['NumGames'] >= 2,
            raw_lineup_wide['GameOrder'],
            0
        ).astype('int')

        del raw_lineup_wide['NumGames']
        del raw_lineup_wide['GameOrder']


        raw_lineup_wide.drop(
                columns=['home_team_name', 'away_team_name'], inplace=True)

        raw_lineup_wide['id'] = np.arange(len(raw_lineup_wide))

        raw_lineup = raw_lineup_wide.melt(
            id_vars=['id', 'home_team_code', 'away_team_code', 'game_time', 'DoubleHeader'],
            var_name='lineup_id',
            value_name='Name')

        del raw_lineup['id']

        # Get First and Last Names
        raw_lineup['FirstName'] = raw_lineup.Name.str.split().str.get(0)
        raw_lineup['FirstName'] = \
            raw_lineup['FirstName'].str.replace('.', '', regex=False)
        raw_lineup['LastName'] = raw_lineup.Name.str.split(n=1).str.get(1)
        raw_lineup['LastName'] = \
            raw_lineup['LastName'].str.replace('.', '', regex=False)
        raw_lineup.drop(columns=['Name'], inplace=True)


        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'M') &
            (raw_lineup['LastName'] == 'Bumgarner'),
            'FirstName'] = 'Madison'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'Michael') &
            (raw_lineup['LastName'] == 'Taylor'),
            'FirstName'] = 'Michael A'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'Hyun') &
            (raw_lineup['LastName'] == 'Jin Ryu'),
            ['FirstName', 'LastName']] = ['Hyun Jin', 'Ryu']

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'D') &
            (raw_lineup['LastName'] == 'Ponce de Leon'),
            ['FirstName', 'LastName']] = ['Daniel', 'Ponce de Leon']

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'A') &
            (raw_lineup['LastName'] == 'DeSclafani'),
            'FirstName'] = 'Anthony'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'J') &
            (raw_lineup['LastName'] == 'Montgomery'),
            'FirstName'] = 'Jordan'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'S') &
            (raw_lineup['LastName'] == 'Strasburg'),
            'FirstName'] = 'Stephen'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'A') &
            (raw_lineup['LastName'] == 'Senzatela'),
            'FirstName'] = 'Antonio'

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'E') &
            (raw_lineup['LastName'] == 'Rodriguez'),
            'FirstName'] = 'Eduardo'

        # set whether players are at home or away, and spot in lineup
        raw_lineup[['home', 'lineup_id']] = \
            raw_lineup['lineup_id'].str.split(pat='_', expand=True)

        # set team variable for each player
        raw_lineup['TEAM'] = raw_lineup['away_team_code']
        raw_lineup.loc[raw_lineup.home == 'home', 'TEAM'] = \
            raw_lineup['home_team_code']

        raw_lineup['base'] = (
            raw_lineup['LastName'].str.replace(r"[\"\', ]", '', regex=True).str[0:4].str.pad(width=4, side='right', fillchar='-').str.lower() +
            raw_lineup['FirstName'].str[0].str.lower() + '0'
        )

        raw_lineup.loc[
            (raw_lineup['FirstName'] == 'Giancarlo') &
            (raw_lineup['LastName'] == 'Stanton'),
            'base'] = 'stanm0'

        raw_lineup['id'] = np.arange(0, raw_lineup.shape[0])

        raw_lineup_x = raw_lineup[['FirstName', 'LastName', 'base', 'id', 'TEAM']]
        raw_lineup_x.columns = ['l_FirstName', 'LastName', 'base', 'l_id', 'l_TEAM']

        test = pd.merge(raw_lineup, roster_match, on=['base', 'LastName', 'FirstName'], how='outer', indicator = True)


        print('Leftovers:')
        print(test.loc[test._merge == 'left_only', ['FirstName', 'LastName', 'base']])

        test['dup_count'] = test.groupby('id')['id'].transform('count')
        test = test.loc[(test.dup_count == 1) | (test.TEAM == test.TEAM_2020)]
        test['dup_count'] = test.groupby('id')['id'].transform('count')

        del test['dup_count']

        lineup_with_id = test.loc[test._merge == 'both']

        lineup_with_id = lineup_with_id[[
            'PLAYER_ID', 'TEAM', 'lineup_id', 'home',
            'home_team_code', 'away_team_code', 'game_time', 'DoubleHeader'
        ]]

        players_grid = lineup_with_id.pivot(
            index=['home_team_code', 'away_team_code', 'game_time', 'DoubleHeader'],
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
            'home_team_code', 'away_team_code', 'game_time', 'DoubleHeader'
        ])
        clean_lineups.columns = [
            'home_pitcher', 'away_pitcher', 'spot', 'BAT_ID'
        ]
        clean_lineups[['home', 'spot']] = \
            clean_lineups['spot'].str.split(pat='_', expand=True)
        clean_lineups['spot'] = clean_lineups['spot'].str.slice(start=-1)

        d = {'home': True, 'away': False}
        clean_lineups['home'] = clean_lineups['home'].map(d)
        clean_lineups

        clean_lineups['PIT_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups['away_pitcher'],
            clean_lineups['home_pitcher']
        )

        clean_lineups['OWN_PIT_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups['home_pitcher'],
            clean_lineups['away_pitcher']
        )

        clean_lineups['PIT_TEAM_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups.index.get_level_values('away_team_code'),
            clean_lineups.index.get_level_values('home_team_code'),
        )

        clean_lineups['BAT_TEAM_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups.index.get_level_values('home_team_code'),
            clean_lineups.index.get_level_values('away_team_code'),
        )

        ## Come back and fix this for double headers
        clean_lineups['GAME_ID'] = \
            clean_lineups.index.get_level_values('home_team_code') + \
            self.__date.replace('-', '') + \
            clean_lineups.index.get_level_values('DoubleHeader').astype('int').astype('str')

        clean_lineups['year'] = int(self.__year)
        clean_lineups = clean_lineups.reset_index()

        clean_lineups = clean_lineups.merge(
            pr2020,
            left_on=['home_team_code'],
            right_on=['team'],
            how='left'
        )

        self.__clean_data = clean_lineups



    def process_data_old(self, output_filepath, stable=True):
        """Process raw data into useful files for model."""

        clean_rosters = Path(output_filepath) / 'rosters.pkl'
        rosters = pd.read_pickle(clean_rosters)
        year_roster = rosters[rosters.year == 2020]

        raw_lineup_wide = self.__raw_data
        recode = {
            'STL': 'SLN',
            'LAD': 'LAN',
            'TB': 'TBA',
            'NYM': 'NYN',
            'CWS': 'CHA',
            'SF': 'SFN',
            'NYY': 'NYA',
            'KC': 'KCA',
            'SD': 'SDN',
            'LAA': 'ANA',
        }
        for old_code, new_code in recode.items():
            raw_lineup_wide.loc[
                raw_lineup_wide.home_team_code == old_code,
                'home_team_code'
            ] = new_code

            raw_lineup_wide.loc[
                raw_lineup_wide.home_team_code == old_code,
                'away_team_code'
            ] = new_code



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
        pairs = indexer.index(raw_lineup, year_roster)

        # Generate matches
        compare = recordlinkage.Compare()
        compare.exact('TEAM', 'TEAM', label='Team')
        compare.exact('LastName', 'LastName', label='LastName')
        compare.string('FirstName', 'FirstName',
                       threshold=thresh, label='FirstName')
        features = compare.compute(pairs, raw_lineup, year_roster)

        # keep best matches
        lineup_with_id = features[features.sum(axis=1) >= 3]
        lineup_with_id = lineup_with_id.reset_index()[['level_0', 'level_1']]

        # merge matches
        lineup_with_id = lineup_with_id.join(
            year_roster, on='level_1', rsuffix='_roster')
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

        clean_lineups['PIT_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups['away_pitcher'],
            clean_lineups['home_pitcher']
        )

        clean_lineups['OWN_PIT_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups['home_pitcher'],
            clean_lineups['away_pitcher']
        )

        clean_lineups['PIT_TEAM_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups.index.get_level_values('away_team_code'),
            clean_lineups.index.get_level_values('home_team_code'),
        )

        clean_lineups['BAT_TEAM_ID'] = np.where(
            clean_lineups['home'] == True,
            clean_lineups.index.get_level_values('home_team_code'),
            clean_lineups.index.get_level_values('away_team_code'),
        )

        ## Come back and fix this for double headers
        clean_lineups['GAME_ID'] = \
            clean_lineups.index.get_level_values('home_team_code') + \
            self.__date.replace('-', '') + '0'

        clean_lineups['year'] = int(self.__year)

        self.__clean_data = clean_lineups

    def write_data(self, output_filepath):
        """Write processed data to directory."""
        save_file = Path(output_filepath) / 'lineup.pkl'
        self.__clean_data.to_pickle(save_file)
