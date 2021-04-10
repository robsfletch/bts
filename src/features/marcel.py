import pandas as pd
from pathlib import Path
import numpy as np


def main_marcel(records, events, people, player_id, metric, base):
    records = records.loc[:, [metric, base, 'G']]
    records = fill_gaps(records, player_id)

    merged = merge_data(records, people, player_id)
    merged[metric] = merged[metric].astype('int')
    merged['G'] = merged['G'].astype('int')

    lr = gen_league(events, metric, base)
    merged['pred_' + metric + 'P' + base], merged['pred_' + base], merged['pred_' + metric] = \
        marcel(merged, lr, player_id, metric, base)

    merged = merged.reset_index()
    merged['year'] = merged['year'] + 1
    merged.set_index([player_id, 'year'])

    merged = merged.rename(columns={'G': 'prev_G'})
    merged = merged.loc[merged[base] != 0]
    return merged

def fill_gaps(records, player_id):
    records = records.reset_index()
    records['first_year'] = records.groupby(player_id)['year'].transform("min")
    records['last_year'] = records.groupby(player_id)['year'].transform("max")

    records = records.set_index([player_id, 'year'])
    records = records.unstack(fill_value=0).stack(dropna=False)
    records['first_year'] = records.groupby(player_id)['first_year'].transform("max")
    records['last_year'] = records.groupby(player_id)['last_year'].transform("max")

    records = records.loc[(records.index.get_level_values('year') <= records['last_year']) &
                  (records.index.get_level_values('year') >= records['first_year'])]
    del records['first_year']
    del records['last_year']

    return records


def merge_data(records, people, player_id):
    records = records.reset_index()

    merged = pd.merge(
        records,
        people['birthYear'],
        left_on = [player_id],
        right_on=['PlayerID'],
        how='left'
    )

    merged = merged.set_index([player_id, 'year']).sort_values([player_id, 'year'])

    merged['Age'] = merged.index.get_level_values('year') - merged['birthYear']
    del merged['birthYear']

    merged = merged.sort_values([player_id, 'year'])

    return merged

def marcel(df, lr, player_id, metric, base):
    w = [5, 4, 3]

    pred_base = est_base(df, player_id, base)
    pred_naive_rate = est_naive_rate(df, w, player_id, metric, base)

    league_adj_rate = reg_to_league_mean(df, pred_naive_rate, w, lr, player_id, base)
    pred_rate = adjust_age(df['Age'], league_adj_rate)

    pred_amt = (pred_base * pred_rate).astype('float')

    return pred_rate, pred_base, pred_amt

def gen_league(events, metric, base):
    League = events[events.BAT_FLD_CD != 1].groupby('year')[[metric, base]].sum()
    League.columns = ['League_count', 'League_base']

    League['L0_League_rate'] = League['League_count'] / League['League_base']
    League['L1_League_rate'] = League['L0_League_rate'].shift(1)
    League['L2_League_rate'] = League['L0_League_rate'].shift(2)
    League = League.sort_values('year')

    return League

def est_base(df, player_id, base):
    group = df.groupby(player_id)[base]
    L0 = group.shift(0).fillna(0)
    L1 = group.shift(1).fillna(0)
    L2 = group.shift(2).fillna(0)

    pred_base = (.5 * L0 + .1 * L1 + 200).astype('float')

    return pred_base

def est_naive_rate(df, w, player_id, metric, base):
    group = df.groupby(player_id)[[metric, base]]
    L0 = group.shift(0).fillna(0)
    L1 = group.shift(1).fillna(0)
    L2 = group.shift(2).fillna(0)

    sum_metric = w[0]*L0[metric] + w[1]*L1[metric] + w[2]*L2[metric]
    sum_base = w[0]*L0[base] + w[1]*L1[base] + w[2]*L2[base]
    naive_rate = sum_metric / sum_base

    return naive_rate

def reg_to_league_mean(df, naive_rate, w, lr, player_id, base):
    temp = pd.merge(df.reset_index(), lr, on='year', how='left')
    temp = temp.set_index([player_id, 'year'])

    group = temp.groupby(player_id)[[base]]
    temp['L0_' + base] = group.shift(0).fillna(0)
    temp['L1_' + base] = group.shift(1).fillna(0)
    temp['L2_' + base] = group.shift(2).fillna(0)

    League_sum = (
        w[0]*temp['L0_League_rate'] +
        w[1]*temp['L1_League_rate'] +
        w[2]*temp['L2_League_rate']
    )

    League_base_weighted_sum = (
        w[0]*temp['L0_League_rate']*temp['L0_' + base] +
        w[1]*temp['L1_League_rate']*temp['L1_' + base] +
        w[2]*temp['L2_League_rate']*temp['L2_' + base]
    )
    sum_base = w[0]*temp['L0_' + base] + w[1]*temp['L1_' + base] + w[2]*temp['L2_' + base]

    League_mean_rate = League_base_weighted_sum / sum_base
    League_mean_rate = League_mean_rate.where(sum_base > 0, League_sum / sum(w))

    reliability = sum_base / (1200 + sum_base)

    adj_rate = League_mean_rate * (1 - reliability) + naive_rate * reliability
    adj_rate = adj_rate.where(sum_base > 0, League_sum / sum(w))

    return adj_rate

def adjust_age(Age, adj_rate):
    age_adj = np.where(
        Age <= 29,
        1 + (29 - Age)* .006,
        1 + (29 - Age)* .003,
    )
    pred_rate = (adj_rate * age_adj).astype('float')

    return pred_rate
