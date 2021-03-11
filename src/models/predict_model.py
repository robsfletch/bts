# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import pickle
import click
import modelsetup

@click.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('selection_file')
def main(data_file, model_file, selection_file):
    with open(model_file, 'rb') as fp:
        fitted_model = pickle.load(fp)

    data = pd.read_pickle(data_file)

    data = data.dropna()
    data = data[data.b_G > 50]

    X, Y = modelsetup.gen_model_data(data)

    probs = fitted_model.predict_proba(X)
    data['EstProb'] = probs[:, 1]
    # probs = fitted_model.predict(X)
    # data['EstProb'] = probs
    # data['EstProb'] = np.random.randint(1, 10000, data.shape[0])
    data = data.set_index(['GAME_ID', 'BAT_ID'])

    selection = data.groupby('Date')['EstProb'].nlargest(2).to_frame()

    selection = selection.sort_values(
        by=['Date', 'EstProb', 'GAME_ID'], ascending=[True, False, True])

    selection['pick_order'] = selection.groupby(['Date']).cumcount()+1
    selection.to_pickle(selection_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
