# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import pickle
import click

@click.command()
@click.argument('interim', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
def main(interim, processed):
    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')
    hits = pd.read_pickle(Path(interim) / 'hits.pkl')
    selections = pd.read_pickle(Path(processed) / 'main_selection.pkl')

    # selection_data = selections.merge(hits, on =['GAME_ID', 'BAT_ID'])
    selection_data = selections.merge(main_data, on =['GAME_ID', 'BAT_ID'])
    selection_data = selection_data.set_index(['Date', 'pick_order'])

    selection_data.to_pickle(Path(processed) / 'selection_data.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
