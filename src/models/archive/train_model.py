# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LogisticRegression, LassoCV,
    LogisticRegressionCV, RidgeCV)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import svm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

import pickle
import click
import modelsetup

from sklearn.metrics import precision_score

@click.command()
@click.argument('processed', type=click.Path(exists=True))
@click.argument('models', type=click.Path(exists=True))
def main(processed, models):
    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')

    main_data = main_data[main_data.b_G > 50]

    train = main_data[(main_data.year > 1990) & (main_data.year < 2018)]
    x_train, y_train, train = modelsetup.gen_model_data(train)

    # fitted_model = LogisticRegression(penalty='l1', solver='liblinear')
    fitted_model = LogisticRegressionCV(cv=10, random_state=0, max_iter=10000)
    # fitted_model = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, random_state=0, max_iter=10000))
    # fitted_model = make_pipeline(StandardScaler(), RidgeCV())
    # fitted_model = svm.SVC(cache_size=8000, probability=True)
    # fitted_model = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', cache_size=8000, probability=True))
    # fitted_model = SGDClassifier(loss='log')
    # fitted_model = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
    # fitted_model = make_pipeline(StandardScaler(), MLPRegressor(random_state=0, max_iter=10000))
    # fitted_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    fitted_model.fit(x_train, y_train.astype('int').values.ravel())

    model_file = Path(models) / 'logistic_model.pkl'
    with open(model_file, 'wb') as fp:
        pickle.dump(fitted_model, fp)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
