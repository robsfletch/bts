# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import os

from sklearn.linear_model import (LogisticRegression, LassoCV,
    LogisticRegressionCV, RidgeCV)
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import (
    PolynomialFeatures, OneHotEncoder, StandardScaler)
from sklearn.compose import ColumnTransformer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.decomposition import PCA

import pickle
import click

from sklearn.metrics import precision_score

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.decomposition import PCA

@click.command()
@click.argument('processed', type=click.Path(exists=True))
@click.argument('models', type=click.Path(exists=True))
def main(processed, models):
    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')

    main_data = main_data[(main_data.b_G > 50)]

    train = main_data[(main_data.year < 2018) & (main_data.year >= 2010)]

    x_vars = [
        'spot', 'home', 'b_HPG', 'p_HPAB', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'p_team_HPAB',
        'p_avg_game_score', 'p_team_avg_game_score'
    ]
    preprocessor =  ColumnTransformer(
        [('spot', 'passthrough', x_vars)],
        remainder='drop'
    )

    fitted_model = make_pipeline(
        preprocessor,
        IterativeImputer(),
        PolynomialFeatures(2, interaction_only=True),
        StandardScaler(),
        LogisticRegressionCV(cv=20, random_state=0, max_iter=10000)
    )
    fitted_model.fit(train, train['Win'].astype('int'))

    model_file = Path(models) / 'logistic_model.pkl'
    with open(model_file, 'wb') as fp:
        pickle.dump(fitted_model, fp)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
