import pandas as pd
from pathlib import Path
import numpy as np

def main_marcel(records, events, team_id, metric, base):
    w = [7, 5, 4, 3]

    events['PIT_TEAM_ID'] = events['HOME_TEAM_ID']
    events.loc[events.BAT_HOME_ID == 1, 'PIT_TEAM_ID'] = events['AWAY_TEAM_ID']

    result = main_marcel(records, events, w, team_id, metric, base)

def main_marcel(records, events, w, team_id, metric, base):
    history = gen_history(ptr, team_id, metric, base)
    team_game_logs = gen_game_log(events, team_id, metric, base)
    league_record = gen_league(events, w, metric, base)

    result = pd.merge(
        team_game_logs.reset_index(),
        history.reset_index(),
        on=[team_id, 'year'],
        how='left'
    )
    result = pd.merge(result, league_record, on=['year'], how='left')

    result['naive_rate'] = est_naive_rate(result, w, metric, base)
    predict = reg_to_league_mean(result, w, metric, base)

    return predict

def gen_history(df, id, metric, base):
    group = df.groupby([id])

    for lag in range(1, 4):
        lag_name = 'L' + str(lag) + '_' + metric
        df[lag_name] = group[metric].shift(lag)
        df[lag_name] = df[lag_name].fillna(0).astype('int')

        lag_name = 'L' + str(lag) + '_' + base
        df[lag_name] = group[base].shift(lag)
        df[lag_name] = df[lag_name].fillna(0).astype('int')

    del df[base]
    del df[metric]

    return df

def gen_game_log(events, id, metric, base):
    bg = events.groupby(['GAME_ID', id]).agg({
        metric: 'sum',
        'Date': 'first',
        'year': 'first',
    })

    bg[base] = 1

    bg = bg.sort_values([id, 'year', 'Date'])

    bg[metric + '_cum'] = bg.groupby([id, 'year'])[metric].cumsum()
    bg[base + '_cum'] = bg.groupby([id, 'year'])[base].cumsum()

    bg[metric + '_cum'] = bg[metric + '_cum'] - bg[metric]
    bg[base + '_cum'] = bg[base + '_cum'] - bg[base]

    return bg

def gen_league(events, w, metric, base):
    League = events.groupby('year').agg({metric: 'sum', 'GAME_ID': 'nunique'})
    League.columns = ['L_count', 'L_base']

    League = League.sort_values('year')
    League['L_rate'] = League['L_count'] / (League['L_base'] * 2)
    League['L1_L_rate'] = League['L_rate'].shift(1)
    League['L2_L_rate'] = League['L_rate'].shift(2)
    League['L3_L_rate'] = League['L_rate'].shift(3)
    League['L_avg_rate'] = (w[1]*League['L1_L_rate'] + w[2]*League['L2_L_rate'] + w[3]*League['L3_L_rate']) / sum(w[1:4])

    del League['L_count']
    del League['L_base']

    return League

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

    reliability = sum_base / (160 + sum_base)

    adj_rate = League_mean_rate * (1 - reliability) + df['naive_rate'] * reliability
    adj_rate = adj_rate.where(sum_base > 0, df['L_avg_rate'])

    return adj_rate
