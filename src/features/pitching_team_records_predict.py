import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    ptr = pd.read_pickle(Path(interim) / 'pitching_team_records.pkl')

    ptr = ptr.sort_values(['PIT_TEAM_ID', 'year'])

    ptr_pred = ptr[['AdjHPG']]
    ptr_pred = ptr_pred.reset_index()
    ptr_pred['year'] = ptr_pred['year']+1
    ptr_pred = ptr_pred.set_index(['PIT_TEAM_ID', 'year'])

    ptr_pred = ptr_pred.rename(columns={
        'AdjHPG': 'p_team_pred_AdjHPG',
    })

    pd.to_pickle(ptr_pred, Path(interim) / 'pitching_team_records_predict.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
