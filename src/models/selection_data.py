# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import pandas as pd
import numpy as np

import pickle
import click

@click.command()
@click.argument('interim', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
@click.option('-e', '--estonly')
def main(interim, processed, estonly = 'False'):

    interim = 'data/interim'
    processed = 'data/processed'

    main_data = pd.read_pickle(Path(processed) / 'main_data.pkl')
    predictions = pd.read_pickle(Path(processed) / 'main_predictions.pkl')

    selections1 = select_by_prob(predictions)

    if estonly == 'True':
        final_selections = selections1
        final_selections['pick_order'] = final_selections.groupby(['Date']).cumcount()+1
        final_selections = final_selections[
            ['GAME_ID', 'BAT_ID', 'EstProb', 'ExpPoints', 'pick_order']
        ]
    else:
        selections2 = select_by_teams(predictions)
        final_selections = combine_selections(selections1, selections2)

    selection_data = final_selections.merge(
        main_data, on =['GAME_ID', 'BAT_ID']
    )

    selections1.to_pickle(Path(processed) / 'selection_by_prob.pkl')
    selection_data.to_pickle(Path(processed) / 'selection_data.pkl')


def select_by_prob(predictions):
    selections = predictions.sort_values(['Date', 'EstProb'], ascending=[True, False])
    selections['pick_order'] = selections.groupby(['Date']).cumcount() + 1
    selections = selections[selections.pick_order <= 2]

    selections['Points'] = selections.groupby('Date')['Win'].transform('prod')
    selections['ExpPoints'] = selections.groupby('Date')['EstProb'].transform('prod')
    selections = selections.reset_index().set_index(['Date', 'pick_order'])

    return selections

def select_by_teams(predictions):
    top_by_team = predictions.sort_values(['Date', 'GAME_ID', 'BAT_TEAM_ID', 'EstProb'], ascending=[True, True, True, False])
    top_by_team['id'] = top_by_team.groupby(['GAME_ID', 'BAT_TEAM_ID']).ngroup()
    top_by_team['pick_order'] = top_by_team.groupby(['GAME_ID', 'BAT_TEAM_ID']).cumcount() + 1

    top_by_team = top_by_team.loc[top_by_team.pick_order <= 2].copy()
    top_by_team['num_pick'] = top_by_team.groupby(['GAME_ID', 'BAT_TEAM_ID'])['pick_order'].transform('max')
    top_by_team = top_by_team.loc[top_by_team.num_pick == 2].copy()

    top_by_team['SD'] = np.sqrt(top_by_team['EstProb'] * (1 - top_by_team['EstProb']))
    top_by_team['Points'] = top_by_team.groupby('id')['Win'].transform('prod')
    top_by_team['ExpPointsPre'] = top_by_team.groupby('id')['EstProb'].transform('prod')
    top_by_team['ProdSD'] = top_by_team.groupby('id')['SD'].transform('prod')
    # top_by_team['ExpPoints'] = top_by_team['ExpPointsPre'] + (.0164 - .0157) + (.9891 - .9728)*top_by_team['ExpPointsPre']
    top_by_team['ExpPoints'] = .03 * (top_by_team['ProdSD']) + top_by_team['ExpPointsPre']


    top_by_team = top_by_team.sort_values(
        ['Date', 'ExpPoints', 'GAME_ID', 'BAT_TEAM_ID', 'EstProb'],
        ascending=[True, False, True, True, False]
    )
    top_by_team['group_pick_order'] = top_by_team.groupby(['Date']).cumcount() + 1

    selections = top_by_team.loc[top_by_team.group_pick_order <= 2].copy()
    selections = selections.reset_index().set_index(['Date', 'pick_order'])
    del selections['group_pick_order']
    del selections['id']
    del selections['ExpPointsPre']
    del selections['num_pick']
    del selections['SD']
    del selections['ProdSD']

    return selections

def combine_selections(selections1, selections2):

    temp1 = selections1.groupby('Date')['ExpPoints'].first().to_frame()
    temp2 = selections2.groupby('Date')['ExpPoints'].first().to_frame()
    temp2.columns = ['ExpPoints2']

    compare = pd.merge(temp1, temp2, on=['Date'], how='left')
    compare['ChooseTeam'] = compare['ExpPoints2'] > compare['ExpPoints']

    selections1 = pd.merge(selections1.reset_index(), compare['ChooseTeam'], on=['Date'], how='left')
    selections2 = pd.merge(selections2.reset_index(), compare['ChooseTeam'], on=['Date'], how='left')

    selections1 = selections1.set_index(['Date', 'pick_order'])
    selections2 = selections2.set_index(['Date', 'pick_order'])

    select1 = selections1[selections1.ChooseTeam == False]
    select2 = selections2[selections2.ChooseTeam == True]
    selections = select1.append(select2)

    selections = selections.sort_values(['Date', 'pick_order'])
    selections = selections[['GAME_ID', 'BAT_ID', 'EstProb', 'ExpPoints', 'ChooseTeam']]

    return selections

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
