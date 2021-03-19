# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LogisticRegression, LassoCV,
    LogisticRegressionCV, RidgeCV, SGDClassifier)
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import (PolynomialFeatures, OneHotEncoder)
from sklearn.compose import ColumnTransformer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

import cloudpickle
import click
import modelsetup

@click.command()
@click.argument('processed', type=click.Path(exists=True))
@click.argument('models', type=click.Path(exists=True))
def main(processed, models):
    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')

    main_data = main_data[(main_data.b_prev_G > 50)]

    train = main_data[(main_data.year < 2000) & (main_data.year >= 1960)]

    fitted_model = model_sgd1()
    fitted_model.fit(train, train['Win'].astype('int'))

    model_file = Path(models) / 'logistic_model.pkl'
    with open(model_file, 'wb') as fp:
        cloudpickle.dump(fitted_model, fp)

def model_nn1():
    x_vars = [
        'spot', 'home', 'b_pred_HPPA', 'p_pred_HPAB', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'own_p_pred_HPAB',
        'p_team_HPG', 'p_team_avg_game_score', 'rating_rating_pre',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'rating_own_rating_pre', 'rating_own_pitcher_rgs'
    ]
    preprocessor =  ColumnTransformer(
        [('spot', 'passthrough', x_vars)],
        remainder='drop'
    )

    callback = EarlyStopping(monitor='loss', patience=2)

    clf = KerasClassifier(
        model=get_clf,
        loss="binary_crossentropy",
        optimizer='adam',
        metrics=[AUC],
        epochs=40,
        batch_size=32,
        callbacks=[callback],
        hidden_layer_dim=8,
        verbose=0,
        optimizer__learning_rate=2e-5,
    )

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('impute', IterativeImputer(random_state = 0)),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    params = {
        "clf__hidden_layer_dim": [5, 8, 12],
        "clf__optimizer__learning_rate": [1e-5, 4e-5, 1e-4, 4e-4]
    }

    gs = GridSearchCV(
        fitted_model, params, refit='AUC', cv=3,
        scoring={'AUC': 'roc_auc'}, n_jobs=-1
    )


    return gs

def get_clf(hidden_layer_dim, meta):
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]

    # model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))

    model = Sequential()
    model.add(Dense(hidden_layer_dim, input_shape=X_shape_[1:]))
    model.add(Activation('relu'))
    model.add(Dense(hidden_layer_dim))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def model_log1():
    x_vars = [
        'spot', 'home', 'b_pred_HPPA', 'p_pred_HPAB', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'own_p_pred_HPAB',
        'p_team_HPG', 'p_team_avg_game_score', 'rating_rating_pre',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'rating_own_rating_pre', 'rating_own_pitcher_rgs'
    ]
    preprocessor =  ColumnTransformer(
        [('spot', 'passthrough', x_vars)],
        remainder='drop'
    )

    clf = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, n_jobs=-1, penalty='l1',
        solver='liblinear'
    )

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('impute', IterativeImputer(random_state = 0)),
        ('poly', PolynomialFeatures(2, interaction_only=True)),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    return fitted_model

def model_sgd1():
    x_vars = [
        'spot', 'home', 'b_pred_HPPA', 'p_pred_HPAB', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND', 'b_avg_win', 'own_p_pred_HPAB',
        'p_team_HPG', 'p_team_avg_game_score', 'rating_rating_pre',
        'rating_rating_prob', 'rating_pitcher_rgs',
        'rating_own_rating_pre', 'rating_own_pitcher_rgs'
    ]
    preprocessor =  ColumnTransformer(
        [('spot', 'passthrough', x_vars)],
        remainder='drop'
    )

    clf = SGDClassifier(loss='log', penalty='l1', random_state=0)

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('impute', IterativeImputer(random_state = 0)),
        ('poly', PolynomialFeatures(2, interaction_only=True)),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    params = {
        "clf__alpha": [.0003, .001, .003, .01]
    }

    gs = GridSearchCV(
        fitted_model, params, refit='AUC', cv=5,
        scoring={'AUC': 'roc_auc'}, n_jobs=-1
    )

    return gs

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
