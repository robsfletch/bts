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
    ev_dir = Path(input_filepath) / 'events'
    all_files = glob.glob(str(ev_dir / "Events*.txt"))

    fields = pd.read_csv(header_file)
    header = fields['Header'].to_numpy()

    types = pd.Series(fields.Type.values,index=fields.Header).to_dict()

    li = []
    for filename in all_files:
        year = int(filename[-8:-4])
        df = pd.read_csv(filename, header=None, names=header, dtype=types,
                         low_memory=False)
        df['year'] = year
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    cols = [
        'BAT_EVENT_FL',
        'LEADOFF_FL', 'PH_FL', 'AB_FL', 'SH_FL', 'SF_FL', 'WP_FL',
        'PB_FL', 'PA_NEW_FL', 'PA_TRUNC_FL', 'PIT_START_FL'
    ]
    for col in cols:
        df[col] = df[col].map({'T': 1, 'F': 0}).astype('Int8')

    df['Date'] = df['GAME_ID'].str.slice(3,11)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    df['H'] = (df['H_FL'] > 0).astype('Int8')
    df['2B'] = (df['H_FL'] == 2).astype('Int8')
    df['3B'] = (df['H_FL'] == 3).astype('Int8')
    df['HR'] = (df['H_FL'] == 4).astype('Int8')
    df['BB'] = (df['EVENT_CD'].isin([14, 15])).astype('Int8')
    df['IW'] = (df['EVENT_CD'] == 15).astype('Int8')
    df['SO'] = (df['EVENT_CD'] == 3).astype('Int8')
    df['HBP'] = (df['EVENT_CD'] == 16).astype('Int8')
    df['BAT_HOME_ID'] = df['BAT_HOME_ID'].astype('Int8')

    df['PA'] = df['AB_FL'] + df['BB'] + df['HBP'] + df['SF_FL'].astype('Int8')

    df.to_pickle(Path(output_filepath) / 'events.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
