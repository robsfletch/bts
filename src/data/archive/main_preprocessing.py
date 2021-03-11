import pandas as pd
import glob
from pathlib import Path


class DataProcessor:
    def __init__(self):
        pass

    def write_data(self, save_file):
        """Write processed data to directory."""
        self.data.to_pickle(save_file)


class GameLogProcessor(DataProcessor):
    def read_data(self, input_filepath):
        """Read raw data into DataProcessor."""
        game_logs = Path(input_filepath) / 'gl1871_2020'
        all_files = glob.glob(str(game_logs / "GL201*.TXT"))
        header_file = Path(input_filepath) / 'game_log_header.csv'

        fields = pd.read_csv(header_file)

        li = []
        for filename in all_files:
            df = pd.read_csv(filename, header=None, names=fields.columns)
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        self.data = df

    def process_data(self, stable=True):
        """Process raw data into useful files for model."""
        self.data['GAME_ID'] = self.data['HomeTeam'] + \
            self.data['Date'].map(str) + \
            self.data['DoubleHeader'].map(str)

        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y%m%d')
        self.data['year'] = self.data.Date.dt.year


class EventsProcessor(DataProcessor):
    def read_data(self, input_filepath):
        """Read raw data into DataProcessor."""
        header_file = Path(input_filepath) / 'fields.csv'
        all_files = glob.glob(str(Path(input_filepath) / "Event201*.txt"))

        fields = pd.read_csv(header_file)
        header = fields['Header'].to_numpy()

        li = []
        for filename in all_files:
            year = int(filename[-8:-4])
            df = pd.read_csv(filename, header=None, names=header,
                             low_memory=False)
            df['year'] = year
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        self.data = df

    def process_data(self, stable=True):
        """Process raw data into useful files for model."""
        self.hits = self.data.groupby(['GAME_ID', 'BAT_ID']).agg({
            'H_FL': 'max',
            'BAT_LINEUP_ID': 'first'
        })
        self.hits['Win'] = self.hits['H_FL'] > 0

    def write_data(self, output_filepath):
        """Write processed data to directory."""
        save_file = Path(output_filepath) / 'events.pkl'
        self.data.to_pickle(save_file)

        clean_hits = Path(output_filepath) / 'batting_games.pkl'
        self.hits.to_pickle(clean_hits)


class RostersProcessor(DataProcessor):
    def read_data(self, input_filepath):
        """Read raw data into DataProcessor."""
        all_files = glob.glob(str(Path(input_filepath) / "2010seve/*.ROS"))

        li = []
        for filename in all_files:
            year = filename[-8:-4]
            df = pd.read_csv(filename, header=None)
            df['year'] = year
            li.append(df)

        df = pd.concat(li, axis=0, ignore_index=True)
        df.columns = [
            'PLAYER_ID', 'LastName', 'FirstName', 'Hand', 'Hand2',
            'TEAM', 'Pos', 'Year'
        ]
        self.data = df

    def process_data(self, stable=True):
        """Process raw data into useful files for model."""
        rosters = self.data
        rosters.LastName = rosters.LastName.str.replace('.', '', regex=False)
        rosters.FirstName = rosters.FirstName.str.replace('.', '', regex=False)

        recode = {
            'CHA': 'CWS',
            'NYA': 'NYY',
            'KCA': 'KC',
            'NYN': 'NYM',
            'CHN': 'CHC',
            'LAN': 'LAD',
            'ANA': 'LAA',
            'SLN': 'STL',
            'TBA': 'TB',
            'WAS': 'WSH',
            'SFN': 'SF',
            'SDN': 'SD',
            }
        for old_code, new_code in recode.items():
            rosters.loc[rosters.TEAM == old_code, 'TEAM'] = new_code

        self.data = rosters
