import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    pr = pd.read_pickle(Path(interim) / 'pitching_records.pkl')

    pr = pr.sort_values(['PIT_ID', 'year'])
    pr_pred = pr[[
        'PIT_HAND', 'G', 'HPPA', 'HPAB'
    ]].groupby('PIT_ID').shift(1)

    # pr_pred = pr_pred.dropna(subset=['G'])
    #
    # pr_pred['w_HPPA'] = pr_pred['HPPA'] * pr_pred['G']
    #
    # pr_pred['sum_w_HPPA'] = pr_pred.groupby('PIT_ID')['w_HPPA'].transform(lambda x: x.rolling(4, 1).mean())
    # pr_pred['sum_G'] = pr_pred.groupby('PIT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    # pr_pred['p_pred_HPPA'] = (pr_pred['sum_w_HPPA'] / pr_pred['sum_G'])
    #
    # del pr_pred['HPPA']
    #
    # pr_pred['w_HPAB'] = pr_pred['HPAB'] * pr_pred['G']
    #
    # pr_pred['sum_w_HPAB'] = pr_pred.groupby('PIT_ID')['w_HPAB'].transform(lambda x: x.rolling(4, 1).mean())
    # pr_pred['sum_G'] = pr_pred.groupby('PIT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    # pr_pred['p_pred_HPAB'] = (pr_pred['sum_w_HPAB'] / pr_pred['sum_G'])
    #
    # del pr_pred['HPAB']

    pr_pred = pr_pred.rename(columns={
        'G': 'p_prev_G',
        'HPAB': 'p_pred_HPAB',
        'HPPA': 'p_pred_HPPA',
    })

    pr_pred = pr_pred.loc[:, [
        'PIT_HAND', 'p_prev_G', 'p_pred_HPAB', 'p_pred_HPPA'
    ]]

    pd.to_pickle(pr_pred, Path(interim) / 'pitching_records_predict.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
