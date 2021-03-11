import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture
def df():
    return pd.read_csv(Path('data/interim') / 'events.pkl')

def test_all_data(df):
    assert df.isna().sum()<1

def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4
