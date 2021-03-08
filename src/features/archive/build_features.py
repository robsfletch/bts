# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import os
from pathlib import Path

from src.data.merge_data import DataMerger


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(interim, processed_filepath):

    data_merger = DataMerger()
    data_merger.merge(interim)
    data_merger.write_data(Path(interim) / 'main.pkl')

    # panel['Win_bin'] = panel['Win']*1
    # panel['cur_avg_win'] = panel.groupby('BAT_ID')['Win_bin'].transform(lambda x: x.rolling(200, 50).mean())
    # panel['avg_win'] = panel.groupby('BAT_ID')['cur_avg_win'].shift(1)
    #
    # # TODO: Pitcher lag needs to lag by game, not by batter
    # panel['cur_pit_avg_win'] = panel.groupby('PitcherID')['Win_bin'].transform(lambda x: x.rolling(100, 50).mean())
    # panel['pit_avg_win'] = panel.groupby('PitcherID')['cur_pit_avg_win'].shift(1)

    ## TODO: Add park factor
    main = main.dropna()

    save_file = processed_data_dir / 'main.pkl'
    main.to_pickle(save_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
