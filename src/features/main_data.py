import pandas as pd
from pathlib import Path
import click
import logging


@click.command()
@click.argument('interim', type=click.Path(exists=True))
@click.argument('processed', type=click.Path(exists=True))
def main(interim, processed):
    merged_data = pd.read_pickle(Path(interim) / 'merged_data.pkl')
    main_data = merged_data

    main_data = main_data[~main_data['Win'].isna()]

    main_data = main_data.astype({
        'home': 'float64',
        'spot': 'float64',
        # 'b_pred_HPG': 'float64',
        'b_pred_AdjHPAdjPA': 'float64',
        # 'p_pred_HPAB': 'float64',
        'p_pred_AdjHPAdjPA': 'float64',
        'park_h_factor': 'float64',
        'year': 'float64',
        'BAT_HAND': 'float64',
        'PIT_HAND': 'float64',
        'opp_hands': 'float64',
        'b_avg_win': 'float64',
        # 'own_p_pred_HPAB': 'float64',
        'own_p_pred_AdjHPAdjPA': 'float64',
        'p_avg_game_score': 'float64',
        'p_own_avg_game_score': 'float64',
        'p_team_pred_AdjHPG': 'float64',
        'p_team_pred_DefEff': 'float64',
        'p_team_avg_game_score': 'float64',
        'b_team_pred_AdjHPG': 'float64',
    })

    main_data.to_pickle(Path(processed) / 'main_data.pkl')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
