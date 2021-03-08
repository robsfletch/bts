import pandas as pd
import glob
from pathlib import Path
import click
import logging

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    header_file = Path(input_filepath) / 'fields.csv'
    all_files = glob.glob(str(Path(input_filepath) / "Event*.txt"))

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

    df.to_pickle(Path(output_filepath) / 'events.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
