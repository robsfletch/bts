# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LogisticRegression, LassoCV,
    LogisticRegressionCV, RidgeCV, SGDClassifier)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (GradientBoostingClassifier,
    HistGradientBoostingClassifier)
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import (PolynomialFeatures, OneHotEncoder)
from sklearn.compose import ColumnTransformer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.calibration import CalibratedClassifierCV

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

    fitted_model = model_nn1()
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

    cv = StratifiedKFold(n_splits=3, shuffle = True, random_state=1)
    gs = GridSearchCV(
        fitted_model, params, refit='AUC', cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=100
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

    cv = StratifiedKFold(shuffle = True, random_state=1)
    clf = LogisticRegressionCV(
        cv=cv, random_state=1, max_iter=10000, n_jobs=-1, penalty='l1',
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

    clf = SGDClassifier(
        loss='log',
        penalty='elasticnet',
        random_state=0,
        max_iter = 1000
    )

    cv_cal = StratifiedKFold(n_splits=2, shuffle = True, random_state=1)
    calibrated_clf = CalibratedClassifierCV(base_estimator = clf, cv=cv_cal, method='isotonic')

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('impute', IterativeImputer(random_state = 0)),
        ('poly', PolynomialFeatures(2, interaction_only=True)),
        ('scale', StandardScaler()),
        ('clf', calibrated_clf),
    ])

    params = {
        "clf__base_estimator__alpha": [.0003, .001, .003, .005, .008, .01],
        # "clf__base_estimator__l1_ratio": [0, .25, .5, .75, 1],
        # "poly__interaction_only": [True, False],
        # "poly__degree": [2, 3]
    }

    cv = StratifiedKFold(shuffle = True, random_state=1)
    gs = HalvingGridSearchCV(
        fitted_model, params, cv=cv, n_jobs=-1, verbose=10,
        refit='roc_auc', scoring='roc_auc',
    )
    # 'roc_auc'
    # scoring={'Precision': 'precision'}

    return gs

def model_xb1():
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


    sgd = SGDClassifier(
        loss='log',
        penalty='l1',
        random_state=0,
        max_iter = 1000,
        alpha = .0005,
    )

    clf = XGBClassifier(
        tree_method='hist',
        verbosity = 0,
        random_state = 0,
        eval_metric = 'auc',
        max_depth = 1,
        learning_rate = .01,
        early_stopping_rounds=5,
        n_estimators=1000,
        use_label_encoder = False,
        # subsample=0.9,
        # colsample_bynode=0.33,
    )

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('impute', IterativeImputer(random_state = 0)),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    params = {
        'clf__max_depth': [1, 2, 4],
        # "clf__n_estimators": [100, 300, 900],
        "clf__learning_rate": [.1, .01, .001]

    }

    cv = StratifiedKFold(shuffle = True, random_state=1)
    gs = HalvingGridSearchCV(
        fitted_model, params, refit='AUC', cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=20
    )


    return gs

def model_xbrf1():
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

    clf = XGBRFClassifier(
        verbosity = 0,
        random_state = 0,
        n_estimators = 500,
    )

    fitted_model = Pipeline([
        ('select', preprocessor),
        ('scale', StandardScaler()),
        ('clf', clf),
    ])

    params = {
        # 'clf__max_depth': [4, 6, 8],
        # "clf__n_estimators": [100, 300, 900],
        "clf__learning_rate": [.1, .01, .001]
    }

    cv = StratifiedKFold(shuffle = True, random_state=1)
    gs = GridSearchCV(
        fitted_model, params, refit='AUC', cv=cv,
        scoring={'AUC': 'roc_auc'}, n_jobs=-1
    )


    return gs

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
