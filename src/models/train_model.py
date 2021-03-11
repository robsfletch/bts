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

from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)

import pickle
import click

from sklearn.metrics import precision_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

@click.command()
@click.argument('processed', type=click.Path(exists=True))
@click.argument('models', type=click.Path(exists=True))
def main(processed, models):
    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')

    main_data = main_data[main_data.b_G > 50]

    train = main_data[(main_data.year > 2000) & (main_data.year < 2018)]
    test = main_data[(main_data.year >= 2018)]

    clf = model9()

    clf.fit(train, train['Win'].astype('int'))
    print("model predicted score: %.3f" % clf.score(train, train['Win'].astype('int')))
    print("model score: %.3f" % clf.score(test, test['Win'].astype('int')))

    model_file = Path(models) / 'logistic_model.pkl'
    with open(model_file, 'wb') as fp:
        pickle.dump(clf, fp)

def model1():
    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home'])],
        remainder='drop'
    )

    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clftype)
    ])

    return clf

def model2():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home']),
        ('b_HPG_scale', numeric_transformer, ['b_HPG'])],
        remainder='drop'
    )

    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clftype)
    ])

    return clf


def model3():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, ['b_HPG', 'p_HPPA'])],
        remainder='drop'
    )

    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clftype)
    ])

    return clf

def model4():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    num_vars = ['b_HPG', 'p_HPPA', 'park_factor']
    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, num_vars)],
        remainder='drop'
    )

    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clftype)
    ])

    return clf

def model5():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    num_vars = ['b_HPG', 'p_HPPA', 'park_factor']
    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, num_vars)],
        remainder='drop'
    )

    poly = PolynomialFeatures(2, interaction_only=True)

    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('classifier', clftype)
    ])

    return clf

def model6():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    num_vars = ['b_HPG', 'p_HPPA', 'park_factor']
    preprocessor =  ColumnTransformer(
        [('spot', StandardScaler(), ['spot']),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, num_vars)],
        remainder='drop'
    )

    poly = PolynomialFeatures(2)
    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('classifier', clftype)
    ])

    return clf

def model7():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    sca_vars = ['spot', 'year']
    num_vars = ['b_HPG', 'p_HPPA', 'park_factor']
    pass_vars = ['BAT_HAND', 'PIT_HAND']
    preprocessor =  ColumnTransformer(
        [('scaled', StandardScaler(), sca_vars),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, num_vars)],
        remainder='drop'
    )

    poly = PolynomialFeatures(2, interaction_only=True)
    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('classifier', clftype)
    ])

    return clf

def model8():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    sca_vars = ['spot', 'year']
    num_vars = [
        'b_HPG', 'p_HPPA', 'park_factor', 'b_avg_win',
        'p_team_HPPA', 'p_avg_game_score', 'p_team_avg_game_score'
    ]
    pass_vars = ['BAT_HAND', 'PIT_HAND']
    preprocessor =  ColumnTransformer(
        [('scaled', StandardScaler(), sca_vars),
        ('home', 'passthrough', ['home']),
        ('past', numeric_transformer, num_vars)],
        remainder='drop'
    )

    poly = PolynomialFeatures(2, interaction_only=True)
    clftype = LogisticRegressionCV(
        cv=5, random_state=0, max_iter=10000, scoring='roc_auc_ovo')
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', poly),
        ('classifier', clftype)
    ])

    return clf


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
