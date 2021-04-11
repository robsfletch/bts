import pandas as pd
from pathlib import Path
import click
import logging
import marcel

@click.command()
@click.argument('interim', type=click.Path(exists=True))
def main(interim):
    pr = pd.read_pickle(Path(interim) / 'pitching_records.pkl')
    events = pd.read_pickle(Path(interim) / 'adj_events.pkl')
    people = pd.read_pickle(Path(interim) / 'people.pkl')

    pr_pred = marcel.main_marcel(pr, events, people, 'PIT_ID', 'AdjH', 'AdjPA')
    pr_pred = pr_pred.rename(columns={
        'pred_AdjHPAdjPA': 'p_pred_AdjHPAdjPA',
        'pred_AdjH': 'p_pred_AdjH',
        'pred_AdjPA': 'p_pred_AdjPA',
        'prev_G': 'p_prev_G',
    })

    pr_pred = pr_pred.loc[:, [
        'p_prev_G', 'p_pred_AdjHPAdjPA', 'p_pred_AdjH', 'p_pred_AdjPA'
    ]]

    pd.to_pickle(pr_pred, Path(interim) / 'pitching_records_predict.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
