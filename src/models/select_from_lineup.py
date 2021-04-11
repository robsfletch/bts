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
        'spot', 'home', 'b_pred_HPPA', 'p_pred_HPPA',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'park_h_factor', 'opp_hands',
        'p_team_pred_AdjHPG', 'p_team_pred_DefEff',
        'b_team_pred_AdjHPG'
    ]

    # data = data.dropna(subset=x_vars)
    data = data[data.b_prev_G > 40]

    probs = fitted_model.predict_proba(data[x_vars])
    data['EstProb'] = probs[:, 1]
    data = data.set_index(['GAME_ID', 'BAT_ID'])

    data = data.sort_values(
        ['EstProb', 'b_pred_HPPA'],
        ascending=[False, False]
    )

    data = data.sort_values(
        by=['EstProb', 'GAME_ID', 'BAT_ID'],
        ascending=[False, True, True]
    )

    data.to_pickle(selection_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
