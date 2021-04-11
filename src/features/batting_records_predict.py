import pandas as pd
from pathlib import Path
import click
import logging
import numpy as np

import marcel

@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    br = pd.read_pickle(Path(interim) / 'batting_records.pkl')
    events = pd.read_pickle(Path(interim) / 'adj_events.pkl')
    people = pd.read_pickle(Path(interim) / 'people.pkl')

    br_pred = marcel.main_marcel(br, events, people, 'BAT_ID', 'AdjH', 'AdjPA')
    br_pred = br_pred.rename(columns={
        'pred_AdjHPAdjPA': 'b_pred_AdjHPAdjPA',
        'pred_AdjH': 'b_pred_AdjH',
        'pred_AdjPA': 'b_pred_AdjPA',
        'prev_G': 'b_prev_G',
    })

    br_pred = br_pred.loc[:, [
        'b_prev_G', 'b_pred_AdjHPAdjPA', 'b_pred_AdjH', 'b_pred_AdjPA'
    ]]

    # br_pred = rolling_avg(br)

    pd.to_pickle(br_pred, Path(interim) / 'batting_records_predict.pkl')

def rolling_avg(br):
    br = br.sort_values(['BAT_ID', 'year'])
    br_pred = br[[
        'BAT_HAND', 'G', 'BBPG', 'ABPG', 'HPG', 'HPPA'
    ]]

    br_pred = br_pred.dropna(subset=['G'])

    br_pred['w_HPPA'] = br_pred['HPPA'] * br_pred['G']

    br_pred['sum_w_HPPA'] = br_pred.groupby('BAT_ID')['w_HPPA'].transform(lambda x: x.rolling(4, 1).mean())
    br_pred['sum_G'] = br_pred.groupby('BAT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    br_pred['b_pred_HPPA'] = (br_pred['sum_w_HPPA'] / br_pred['sum_G'])

    del br_pred['HPPA']

    br_pred['w_HPG'] = br_pred['HPG'] * br_pred['G']

    br_pred['sum_w_HPG'] = br_pred.groupby('BAT_ID')['w_HPG'].transform(lambda x: x.rolling(4, 1).mean())
    br_pred['sum_G'] = br_pred.groupby('BAT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    br_pred['b_pred_HPG'] = (br_pred['sum_w_HPG'] / br_pred['sum_G'])

    del br_pred['HPG']

    br_pred = br_pred.rename(columns={
        'G': 'b_prev_G',
        'BBPG': 'b_pred_BBPG',
        'ABPG': 'b_pred_ABPG',
    })

    br_pred = br_pred.loc[:, [
        'BAT_HAND', 'b_prev_G', 'b_pred_BBPG', 'b_pred_ABPG',
        'b_pred_HPG', 'b_pred_HPPA'
    ]]

    br_pred = br_pred.reset_index()
    br_pred['year'] = br_pred['year'] + 1
    br_pred.set_index(['BAT_ID', 'year'])

    return br_pred



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
