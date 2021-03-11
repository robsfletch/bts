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

    X, Y, row_list = model2(df)
    return X, Y, row_list


def model1(df):

    x_vars = ['spot', 'home']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    return X, Y, row_list

def model2(df):

    x_vars = ['spot', 'home', 'b_HPG']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler())
    ])
    X = mapper.fit_transform(X)

    return X, Y, row_list

def model3(df):
    x_vars = ['spot', 'home', 'b_HPG', 'p_HPPA']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    return X, Y, row_list

def model4(df):
    x_vars = ['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    return X, Y, row_list

def model5(df):
    x_vars = ['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    return X, Y, row_list

def model6(df):
    x_vars = ['spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor']
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)

    return X, Y, row_list

def model7(df):
    x_vars = [
        'spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND'
    ]
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
        (['year'], StandardScaler()),
        (['BAT_HAND'], StandardScaler()),
        (['PIT_HAND'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    return X, Y, row_list

def model8(df):
    x_vars = [
        'spot', 'home', 'b_HPG', 'p_HPPA', 'park_factor', 'year',
        'BAT_HAND', 'PIT_HAND'
    ]
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

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

    X = mapper.fit_transform(X)
    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    return X, Y, row_list

def model9(df):
    x_vars = [
        'spot', 'home', 'b_HPG', 'b_ABPG', 'b_BBPG', 'p_HPPA',
        'park_factor', 'year', 'BAT_HAND', 'PIT_HAND', 'b_avg_win',
        'p_team_HPPA', 'p_avg_game_score', 'p_team_avg_game_score'
    ]
    y_var = ['Win']
    vars = x_vars + y_var

    df[vars].isnull().any(axis=1)
    row_list = df[vars].isnull().any(axis=1)
    X = df.loc[~row_list, x_vars]
    Y = df.loc[~row_list, y_var]

    mapper = DataFrameMapper([
        (['spot'], StandardScaler()),
        (['home'], StandardScaler()),
        (['b_HPG'], StandardScaler()),
        (['p_HPPA'], StandardScaler()),
        (['park_factor'], StandardScaler()),
        (['year'], StandardScaler()),
        (['BAT_HAND'], StandardScaler()),
        (['PIT_HAND'], StandardScaler()),
    ])
    X = mapper.fit_transform(X)

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    return X, Y, row_list
