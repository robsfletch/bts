import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    ptr = pd.read_pickle(Path(interim) / 'pitching_records.pkl')

    ptr = ptr.sort_values(['PIT_TEAM_ID', 'year'])

    ######## STOPPED HERE

    ptr_pred = ptr[[
        'PIT_HAND', 'G', 'HPPA', 'HPAB'
    ]].groupby('PIT_ID').shift(1)

    ptr_pred = ptr_pred.dropna(subset=['G'])

    ptr_pred['w_HPPA'] = ptr_pred['HPPA'] * ptr_pred['G']

    ptr_pred['sum_w_HPPA'] = ptr_pred.groupby('PIT_ID')['w_HPPA'].transform(lambda x: x.rolling(4, 1).mean())
    ptr_pred['sum_G'] = ptr_pred.groupby('PIT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    ptr_pred['p_pred_HPPA'] = (ptr_pred['sum_w_HPPA'] / ptr_pred['sum_G'])

    del ptr_pred['HPPA']

    ptr_pred['w_HPAB'] = ptr_pred['HPAB'] * ptr_pred['G']

    ptr_pred['sum_w_HPAB'] = ptr_pred.groupby('PIT_ID')['w_HPAB'].transform(lambda x: x.rolling(4, 1).mean())
    ptr_pred['sum_G'] = ptr_pred.groupby('PIT_ID')['G'].transform(lambda x: x.rolling(4, 1).mean())
    ptr_pred['p_pred_HPAB'] = (ptr_pred['sum_w_HPAB'] / ptr_pred['sum_G'])

    del ptr_pred['HPAB']

    ptr_pred = ptr_pred.rename(columns={
        'G': 'p_prev_G',
    })

    ptr_pred = ptr_pred.loc[:, [
        'PIT_HAND', 'p_prev_G', 'p_pred_HPAB', 'p_pred_HPPA'
    ]]

    pd.to_pickle(ptr_pred, Path(interim) / 'pitching_records_predict.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
