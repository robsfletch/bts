import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    ptr = pd.read_pickle(Path(interim) / 'pitching_team_records.pkl')

    ptr = ptr.sort_values(['PIT_TEAM_ID', 'year'])
    ptr_pred = ptr.loc[:, ['AdjH', 'G']]
    ptr_pred['p_team_pred_AdjHPG'] = weighted_avg(ptr, 'PIT_TEAM_ID', 'AdjH', 'G')
    ptr_pred['p_team_pred_DefEff'] = weighted_avg(ptr, 'PIT_TEAM_ID', 'DefEff', 'G')


    ptr_pred = ptr[['AdjHPG', 'DefEff']]
    ptr_pred = ptr_pred.reset_index()
    ptr_pred['year'] = ptr_pred['year']+1
    ptr_pred = ptr_pred.set_index(['PIT_TEAM_ID', 'year'])

    ptr_pred = ptr_pred.rename(columns={
        'AdjHPG': 'p_team_pred_AdjHPG',
        'DefEff': 'p_team_pred_DefEff',
    })

    pd.to_pickle(ptr_pred, Path(interim) / 'pitching_team_records_predict.pkl')

def weighted_avg(df, id, metric, base):
    df = df.sort_values([id, 'year'])

    sum_metric = df.groupby(id)[metric].transform(lambda x: x.rolling(4, 1).sum())
    sum_base = df.groupby(id)[base].transform(lambda x: x.rolling(4, 1).sum())
    avg = (sum_metric / sum_base)

    return avg



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
