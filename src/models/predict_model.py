# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import cloudpickle
import click

@click.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('selection_file')
def main(data_file, model_file, selection_file):
    with open(model_file, 'rb') as fp:
        fitted_model = cloudpickle.load(fp)

    data = pd.read_pickle(data_file)

    x_vars = [
        'spot', 'home', 'b_pred_HPPA', 'p_HPAB', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'p_own_HPAB',
        'p_team_HPAB', 'p_team_avg_game_score', 'rating_rating_pre',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'rating_own_rating_pre', 'rating_own_pitcher_rgs'
    ]
    data = data.dropna(subset=x_vars)
    data = data[data.b_prev_G > 50]

    probs = fitted_model.predict_proba(data)
    data['EstProb'] = probs[:, 1]
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
