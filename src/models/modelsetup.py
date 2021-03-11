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

    X = df[['spot', 'home', 'b_HPG']]
    Y = df['Win']

    return X, Y

def model3(df):
    X = df[['spot', 'home', 'b_HPG', 'p_HPPA']]
    Y = df['Win']

    return X, Y

def model4(df):
    X = df[['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']]
    Y = df['Win']

    return X, Y

def model5(df):
    X = df[['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']]

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model6(df):
    X = df[['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']]

    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model7(df):
    X = df[['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor', 'year', 'BAT_HAND', 'PIT_HAND']] # , 'avg_win'

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model8(df):
    mapper = DataFrameMapper([
        (['spot'], OneHotEncoder(drop='first')),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
        (['year'], StandardScaler()),
        (['BAT_HAND'], OneHotEncoder(drop='first')),
        (['PIT_HAND'], OneHotEncoder(drop='first')),
    ])

    X = mapper.fit_transform(df.copy())
    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y

def model9(df):
    X = df[['spot', 'home', 'b_HPG', 'b_ABPG', 'b_BBPG', 'p_HPPA',
    'park_factor', 'year', 'BAT_HAND', 'PIT_HAND', 'b_avg_win',
    'p_team_HPPA', 'p_avg_game_score', 'p_team_avg_game_score']] #

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y
