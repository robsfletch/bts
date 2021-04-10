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
@click.argument('prediction_file')
def main(data_file, model_file, prediction_file):
    with open(model_file, 'rb') as fp:
        fitted_model = cloudpickle.load(fp)

    data = pd.read_pickle(data_file)

    x_vars = [
        'spot', 'home', 'b_pred_HPPA', 'p_pred_HPPA',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'park_h_factor', 'opp_hands',
        'p_team_pred_AdjHPG'
    ]
    y_var = ['Win']
    vars = x_vars + y_var
    # x_vars = [
    #     'spot', 'home', 'b_pred_HPPA', 'p_pred_HPAB', 'park_factor', 'year',
    #     'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'own_p_pred_HPAB',
    #     'p_team_HPAB', 'p_team_avg_game_score', 'rating_rating_pre',
    #     'rating_rating_prob', 'rating_pitcher_rgs',
    #     'rating_own_rating_pre', 'rating_own_pitcher_rgs'
    # ]
    data = data.dropna(subset=x_vars)
    data = data.loc[data.b_prev_G > 50]


    probs = fitted_model.predict_proba(data[x_vars])
    data['EstProb'] = probs[:, 1]
    data = data.set_index(['GAME_ID', 'BAT_ID'])

    data = data.sort_values(
        ['EstProb', 'b_pred_HPPA', 'b_avg_win'],
        ascending=[False, False, False]
    )

    data.to_pickle(prediction_file)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
