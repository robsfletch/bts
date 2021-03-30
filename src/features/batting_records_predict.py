import pandas as pd
from pathlib import Path
import click
import logging
import numpy as np


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    br = pd.read_pickle(Path(interim) / 'batting_records.pkl')
    gl = pd.read_pickle(Path(interim) / 'game_logs.pkl')
    events = pd.read_pickle(Path(interim) / 'events.pkl')
    people = pd.read_pickle(Path(interim) / 'people.pkl')

    # br_pred = marcel(br, gl, events, people)


    br_pred = rolling_avg(br)

    pd.to_pickle(br_pred, Path(interim) / 'batting_records_predict.pkl')

def rolling_avg(br):
    br = br.sort_values(['BAT_ID', 'year'])
    br_pred = br[[
        'BAT_HAND', 'G', 'BBPG', 'ABPG', 'HPG', 'HPPA'
    ]].groupby('BAT_ID').shift(1)

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

    return br_pred

def marcel(br, gl, events, people):
    ## League DAta
    League = events[events.BAT_FLD_CD != 1].groupby('year')[['HR', 'H', 'PA']].sum()
    League.columns = ['League_HR', 'League_H', 'League_PA']

    League['League_HPPA'] = League['League_H'] / League['League_PA']
    League = League.sort_values('year')
    League['exp_League_HPPA'] = (
        5 * League['League_HPPA'].shift(1) +
        4 * League['League_HPPA'].shift(2) +
        3 * League['League_HPPA'].shift(3)
    )/12
    League['exp_League_HPPA'] = League['exp_League_HPPA'].astype('float')

    ## combine data
    br = br.reset_index()
    merged = pd.merge(br, League, on=['year'], how='left')
    merged = pd.merge(merged, people[['PlayerID', 'birthYear']], left_on = ['BAT_ID'], right_on=['PlayerID'], how='left')
    merged = merged.set_index(['BAT_ID', 'year']).sort_values(['BAT_ID', 'year'], ascending=[True, True])

    merged['Age'] = merged.index.get_level_values('year') - merged['birthYear']

    marcel = merged
    marcel = marcel.sort_values(['BAT_ID', 'year'])

    metric = 'H'

    group = marcel.groupby('BAT_ID')
    marcel['L1_' + metric] = group[metric].shift(1, fill_value=0)
    marcel['L2_' + metric] = group[metric].shift(2, fill_value=0)
    marcel['L3_' + metric] = group[metric].shift(3, fill_value=0)

    marcel['L1_PA'] = group['PA'].shift(1, fill_value=0)
    marcel['L2_PA'] = group['PA'].shift(2, fill_value=0)
    marcel['L3_PA'] = group['PA'].shift(3, fill_value=0)

    marcel['League_' + metric + '_PPA'] = marcel['League_' + metric] / marcel['League_PA']
    marcel['L1_League_' + metric + '_PPA'] = group['League_' + metric + '_PPA'].shift(1, fill_value=0)
    marcel['L2_League_' + metric + '_PPA'] = group['League_' + metric + '_PPA'].shift(2, fill_value=0)
    marcel['L3_League_' + metric + '_PPA'] = group['League_' + metric + '_PPA'].shift(3, fill_value=0)

    marcel['exp_' + metric] = marcel['PA'] * marcel['League_' + metric + '_PPA']
    marcel['L1_exp_' + metric] = group['exp_' + metric].shift(1, fill_value=0)
    marcel['L2_exp_' + metric] = group['exp_' + metric].shift(2, fill_value=0)
    marcel['L3_exp_' + metric] = group['exp_' + metric].shift(3, fill_value=0)

    marcel['exp_PA'] = .5 * marcel['L1_PA'] + .1 * marcel['L2_PA'] + 200

    marcel['sum_' + metric] = (
        5 * marcel['L1_' + metric] +
        4 * marcel['L2_' + metric] +
        3 * marcel['L3_' + metric]
    )
    marcel['sum_PA'] = (
        5 * marcel['L1_PA'] +
        4 * marcel['L2_PA'] +
        3 * marcel['L3_PA']
    )
    marcel['exp_sum_' + metric] = (
        5 * marcel['L1_exp_' + metric] +
        4 * marcel['L2_exp_' + metric] +
        3 * marcel['L3_exp_' + metric]
    )

    marcel['exp_' + metric + '_per_1200'] = (marcel['exp_sum_' + metric] / marcel['sum_PA']) * 1200
    marcel['exp_' + metric + '_per_pa'] = np.where(
        marcel['sum_PA'] > 0,
        (marcel['exp_' + metric + '_per_1200'] + marcel['sum_' + metric]) / (1200 +  marcel['sum_PA']),
        marcel['exp_League_' + metric + 'PPA']
    ).astype('float')

    marcel['age_adj'] = np.where(
        marcel['Age'] <= 29,
        1 + (29 - marcel['Age'])* .006,
        1 + (29 - marcel['Age'])* .003,
    )

    marcel['adj_exp_' + metric + '_per_pa'] = marcel['exp_' + metric + '_per_pa'] * marcel['age_adj']
    marcel['adj_exp_' + metric] = marcel['adj_exp_' + metric + '_per_pa'] *  marcel['exp_PA']

    merged['pred_HPPA'] = marcel['adj_exp_H_per_pa'].astype('float')

    merged = merged.sort_values(['BAT_ID', 'year'])
    merged['prev_G'] = merged.groupby('BAT_ID')['G'].shift(1)

    br_pred = merged[['BAT_HAND', 'pred_HPPA', 'prev_G']]

    br_pred = br_pred.rename(columns={
        'prev_G': 'b_prev_G',
        'pred_HPPA': 'b_pred_HPPA',
    })

    return br_pred





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
