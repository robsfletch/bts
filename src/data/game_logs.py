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
    df = read_data(input_filepath)
    processed_df = process_data(df)
    processed_df.to_pickle(Path(output_filepath) / 'game_logs.pkl')

def read_data(input_filepath):
    """Read raw data into DataProcessor."""
    game_logs = Path(input_filepath) / 'gl1871_2020'
    all_files = glob.glob(str(game_logs / "GL*.TXT"))
    header_file = Path(input_filepath) / 'game_log_header.csv'

    fields = pd.read_csv(header_file)
    header = fields['Header'].to_numpy()

    types = pd.Series(fields.Type.values,index=fields.Header).to_dict()


    li = []
    for filename in all_files:
        df = pd.read_csv(filename, header=None, names=header, dtype=types,
                         low_memory=False)
        df = df.drop(columns=df.filter(regex='.*Name$').columns)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def process_data(df, stable=True):
    """Process raw data into useful files for model."""
    df['GAME_ID'] = df['HomeTeam'] + df['Date'].map(str) + \
        df['DoubleHeader'].map(str)

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['year'] = df.Date.dt.year

    df['HomePA']  = (df['HomeAB'] + df['HomeBB'] +
        df['HomeHBP'] + df['HomeSF'] + df['HomeSH'])
    df['VisitorPA']  = (df['VisitorAB'] + df['VisitorBB']
        + df['VisitorHBP'] + df['VisitorSF'] + df['VisitorSH'])

    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
