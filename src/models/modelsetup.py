import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    LabelBinarizer,
    StandardScaler,
)
from sklearn_pandas import DataFrameMapper

def gen_model_data(df):

    X, Y = model9(df)
    return X, Y


def model1(df):

    X = df[['spot', 'home']]
    Y = df['Win']

    return X, Y

def model2(df):

    X = df[['spot', 'home', 'HPG']]
    Y = df['Win']

    return X, Y

def model3(df):
    X = df[['spot', 'home', 'HPG', 'HPAB_p']]
    Y = df['Win']

    return X, Y

def model4(df):
    X = df[['spot', 'home', 'HPG', 'HPAB_p', 'factor']]
    Y = df['Win']

    return X, Y

def model5(df):
    X = df[['spot', 'home', 'HPG', 'HPAB_p', 'factor']]

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model6(df):
    X = df[['spot', 'home', 'HPG', 'HPAB_p', 'factor']]

    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model7(df):
    X = df[['spot', 'home', 'HPG', 'HPAB_p', 'factor', 'year', 'BAT_HAND', 'PIT_HAND']] # , 'avg_win'

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model8(df):
    mapper = DataFrameMapper([
        (['spot'], OneHotEncoder(drop='first')),
        (['home'], None),
        (['HPG'], StandardScaler()),
        (['HPAB_p'], StandardScaler()),
        (['factor'], StandardScaler()),
        (['year'], StandardScaler()),
        (['BAT_HAND'], None),
        (['PIT_HAND'], None),
    ])

    X = mapper.fit_transform(df.copy())
    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model9(df):
    X = df[['spot', 'home', 'HPG', 'ABPG', 'BBPG', 'HPAB_p',
    'factor', 'year', 'BAT_HAND', 'PIT_HAND', 'avg_win',
    'HPAB_p_team', 'avg_game_score', 'avg_game_score_team']] #

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y
