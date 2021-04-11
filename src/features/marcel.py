import pandas as pd
from pathlib import Path
import numpy as np


def main_marcel(records, events, people, player_id, metric, base):
    records = records.loc[:, [metric, base, 'G']]
    records = fill_gaps(records, player_id)

    w = [7, 5, 4, 3]

    result = marcel(records, events, people, w, player_id, metric, base)

    result = result.rename(columns={'G': 'prev_G'})
    return result

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

def marcel(records, events, people, w, player_id, metric, base):
    full_records = add_age(records, people, player_id)
    full_records[metric] = full_records[metric].astype('int')
    full_records = full_records.sort_values([player_id, 'year'])

    history = gen_history(full_records, player_id, metric, base)
    player_game_logs = gen_player_log(events, player_id, metric, base)
    league_record = gen_league(events, w, metric, base)

    result = pd.merge(
        player_game_logs.reset_index(),
        history.reset_index(),
        on=[player_id, 'year'],
        how='left'
    )
    result = pd.merge(result, league_record, on=['year'], how='left')

    result['pred_' + base] = est_base(result, base)
    result['naive_rate'] = est_naive_rate(result, w, metric, base)
    result['adj_rate'] = reg_to_league_mean(result, w, metric, base)
    result['pred_' + metric + 'P' + base] = adjust_age(result)

    result['pred_' + metric] = (result['pred_' + base] * result['pred_' + metric + 'P' + base]).astype('float')

    result = result.set_index(['GAME_ID', player_id])
    result = result[[
        'pred_' + metric + 'P' + base,
        'pred_' + metric,
        'pred_' + base,
        'G'
    ]]

    return result

def add_age(records, people, player_id):
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

def gen_history(merged, player_id, metric, base):
    group = merged.groupby([player_id])

    for lag in range(1, 4):
        lag_name = 'L' + str(lag) + '_' + metric
        merged[lag_name] = group[metric].shift(lag)
        merged[lag_name] = merged[lag_name].fillna(0).astype('int')

        lag_name = 'L' + str(lag) + '_' + base
        merged[lag_name] = group[base].shift(lag)
        merged[lag_name] = merged[lag_name].fillna(0).astype('int')

    del merged[base]
    del merged[metric]

    return merged

def gen_player_log(events, player_id, metric, base):
    bg = events.groupby(['GAME_ID', player_id]).agg({
        metric: 'sum',
        base: 'sum',
        'Date': 'first',
        'year': 'first',
    })

    bg = bg.sort_values([player_id, 'year', 'Date'])

    bg[metric + '_cum'] = bg.groupby([player_id, 'year'])[metric].cumsum()
    bg[base + '_cum'] = bg.groupby([player_id, 'year'])[base].cumsum()

    bg[metric + '_cum'] = bg[metric + '_cum'] - bg[metric]
    bg[base + '_cum'] = bg[base + '_cum'] - bg[base]

    return bg

def gen_league(events, w, metric, base):
    League = events[events.BAT_FLD_CD != 1].groupby('year')[[metric, base]].sum()
    League.columns = ['L_count', 'L_base']

    League = League.sort_values('year')
    League['L_rate'] = League['L_count'] / League['L_base']
    League['L1_L_rate'] = League['L_rate'].shift(1)
    League['L2_L_rate'] = League['L_rate'].shift(2)
    League['L3_L_rate'] = League['L_rate'].shift(3)
    League['L_avg_rate'] = (w[1]*League['L1_L_rate'] + w[2]*League['L2_L_rate'] + w[3]*League['L3_L_rate']) / sum(w[1:4])

    del League['L_count']
    del League['L_base']

    return League

def est_base(df, base):
    pred_base = (.5 * df['L1_' + base] + .1 * df['L2_' + base] + 200).astype('float')

    return pred_base

def est_naive_rate(df, w, metric, base):
    sum_metric = w[0]*df[metric + '_cum'] + w[1]*df['L1_' + metric] + w[2]*df['L2_' + metric] + w[3]*df['L3_' + metric]
    sum_base = w[0]*df[base + '_cum'] + w[1]*df['L1_' + base] + w[2]*df['L2_' + base] + w[3]*df['L3_' + base]
    naive_rate = sum_metric / sum_base

    return naive_rate

def reg_to_league_mean(df, w, metric, base):
    League_base_weighted_sum = (
        w[1]*df['L1_L_rate']*df['L1_' + base] +
        w[2]*df['L2_L_rate']*df['L2_' + base] +
        w[3]*df['L3_L_rate']*df['L3_' + base]
    )

    sum_base = w[1]*df['L1_' + base] + w[2]*df['L2_' + base] + w[3]*df['L3_' + base]

    League_mean_rate = League_base_weighted_sum / sum_base
    League_mean_rate = League_mean_rate.where(sum_base > 0, df['L_avg_rate'])

    reliability = sum_base / (1200 + sum_base)

    adj_rate = League_mean_rate * (1 - reliability) + df['naive_rate'] * reliability
    adj_rate = adj_rate.where(sum_base > 0, df['L_avg_rate'])

    return adj_rate

def adjust_age(df):
    age_adj = np.where(
        df.Age <= 29,
        1 + (29 - df.Age)* .006,
        1 + (29 - df.Age)* .003,
    )
    pred_rate = (df['adj_rate'] * age_adj).astype('float')

    return pred_rate
