import pandas as pd
import glob
from pathlib import Path
import click
import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """clean game_logs"""
    rosters = read_data(input_filepath)
    clean_rosters = process_data(rosters)
    clean_rosters.to_pickle(Path(output_filepath) / 'rosters.pkl')

def read_data(input_filepath):
    """Read raw data into DataProcessor."""
    all_files = glob.glob(str(Path(input_filepath) / "Rosters/*.ROS"))

    li = []
    for filename in all_files:
        year = int(filename[-8:-4])
        df = pd.read_csv(filename, header=None)
        df['year'] = year
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df.columns = [
        'PLAYER_ID', 'LastName', 'FirstName', 'BattingHand', 'ThrowingHand',
        'TEAM', 'Pos', 'year'
    ]

    return df

def process_data(df, stable=True):
    """Process raw data into useful files for model."""
    df.LastName = df.LastName.str.replace('.', '', regex=False)
    df.FirstName = df.FirstName.str.replace('.', '', regex=False)
    # 
    # recode = {
    #     'CHA': 'CWS',
    #     'NYA': 'NYY',
    #     'KCA': 'KC',
    #     'NYN': 'NYM',
    #     'CHN': 'CHC',
    #     'LAN': 'LAD',
    #     'ANA': 'LAA',
    #     'SLN': 'STL',
    #     'TBA': 'TB',
    #     'WAS': 'WSH',
    #     'SFN': 'SF',
    #     'SDN': 'SD',
    #     }
    # for old_code, new_code in recode.items():
    #     df.loc[df.TEAM == old_code, 'TEAM'] = new_code

    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
