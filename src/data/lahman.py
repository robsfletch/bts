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
    people = read_data(input_filepath)
    clean_people = process_data(people)
    clean_people.to_pickle(Path(output_filepath) / 'people.pkl')

def read_data(input_filepath):
    """Read raw data into DataProcessor."""
    df = pd.read_csv(Path(input_filepath) / 'People.csv')

    df = df.rename(columns={'playerID':'lahmanID'})
    df = df.rename(columns={'retroID':'PlayerID'})

    return df




def process_data(df, stable=True):
    """Process raw data into useful files for model."""
    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
