import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    btr = pd.read_pickle(Path(interim) / 'batting_team_records.pkl')

    btr = btr.sort_values(['BAT_TEAM_ID', 'year'])

    btr_pred = btr[['AdjHPG']]
    btr_pred = btr_pred.reset_index()
    btr_pred['year'] = btr_pred['year']+1
    btr_pred = btr_pred.set_index(['BAT_TEAM_ID', 'year'])

    btr_pred = btr_pred.rename(columns={
        'AdjHPG': 'b_team_pred_AdjHPG',
    })

    pd.to_pickle(btr_pred, Path(interim) / 'batting_team_records_predict.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
