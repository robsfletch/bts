import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.preprocessing import (PolynomialFeatures, OneHotEncoder)

def gen_model_data(df):

    X, Y = model7(df)
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
    X = df[['spot', 'home', 'HPG', 'HPAB_p', 'factor', 'year', 'BAT_HAND', 'PIT_HAND']]

    poly = PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)

    Y = df['Win']

    return X, Y


    # poly = PolynomialFeatures(2, interaction_only=True)
    # X = df[['spot', 'home', 'HPG']]  # , 'HPG' , 'avg_win', 'pit_avg_win'

    # enc = OneHotEncoder(dtype=np.int, drop='first', sparse=True)
    # X = pd.DataFrame(
    #     enc.fit_transform(df.spot.values.reshape(-1, 1)).toarray(),
    #     columns=['spot_2', 'spot_3', 'spot_4', 'spot_5',
    #         'spot_6', 'spot_7', 'spot_8', 'spot_9']
    #     )

    # X['home'] = df['home']

    # X = poly.fit_transform(X)
    # Y = df['Win']

    # return X, Y
