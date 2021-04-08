import pandas as pd
import glob
from pathlib import Path
import click
import logging
from src import utils
import os



def main():
    """clean game_logs"""
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = utils.get_raw_data(project_dir)
    interim_data_dir = utils.get_interim_data(project_dir)

    df = read_data(raw_data_dir)
    processed_df = process_data(df)
    processed_df.to_pickle(Path(interim_data_dir) / 'game_logs.pkl')

    return df

def read_data(raw_data_dir):
    """Read raw data into DataProcessor."""
    ev_dir = Path(raw_data_dir) / 'events'
    all_files = glob.glob(str(ev_dir / "Games*.txt"))


    dtypes = {
        'GAME_CT': 'Int8',
        'METHOD_RECORD_CD': 'Int8',
        'PITCHES_RECORD_CD': 'Int8',
        'TEMP_PARK_CT': 'Int8',
        'WIND_DIRECTION_PARK_CD': 'Int8',
        'WIND_SPEED_PARK_CT': 'Int8',
        'FIELD_PARK_CD': 'Int8',
        'PRECIP_PARK_CD': 'Int8',
        'SKY_PARK_CD': 'Int8',
        'MINUTES_GAME_CT': 'Int16',
        'INN_CT': 'Int8',
        'AWAY_SCORE_CT': 'Int8',
        'HOME_SCORE_CT': 'Int8',
        'AWAY_HITS_CT': 'Int8',
        'HOME_HITS_CT': 'Int8',
        'AWAY_ERR_CT': 'Int8',
        'HOME_ERR_CT': 'Int8',
        'AWAY_LOB_CT': 'Int8',
        'HOME_LOB_CT': 'Int8',
        'AWAY_LINEUP1_FLD_CD': 'Int8',
        'AWAY_LINEUP2_FLD_CD': 'Int8',
        'AWAY_LINEUP3_FLD_CD': 'Int8',
        'AWAY_LINEUP4_FLD_CD': 'Int8',
        'AWAY_LINEUP5_FLD_CD': 'Int8',
        'AWAY_LINEUP6_FLD_CD': 'Int8',
        'AWAY_LINEUP7_FLD_CD': 'Int8',
        'AWAY_LINEUP8_FLD_CD': 'Int8',
        'AWAY_LINEUP9_FLD_CD': 'Int8',
        'HOME_LINEUP1_FLD_CD': 'Int8',
        'HOME_LINEUP2_FLD_CD': 'Int8',
        'HOME_LINEUP3_FLD_CD': 'Int8',
        'HOME_LINEUP4_FLD_CD': 'Int8',
        'HOME_LINEUP5_FLD_CD': 'Int8',
        'HOME_LINEUP6_FLD_CD': 'Int8',
        'HOME_LINEUP7_FLD_CD': 'Int8',
        'HOME_LINEUP8_FLD_CD': 'Int8',
        'HOME_LINEUP9_FLD_CD': 'Int8',
        'AWAY_TEAM_GAME_CT': 'Int16',
        'HOME_TEAM_GAME_CT': 'Int16',
        'OUTS_CT': 'Int16',
        'AWAY_AB_CT': 'Int16',
        'AWAY_2B_CT': 'Int16',
        'AWAY_3B_CT': 'Int16',
        'AWAY_HR_CT': 'Int16',
        'AWAY_BI_CT': 'Int16',
        'AWAY_SH_CT': 'Int16',
        'AWAY_SF_CT': 'Int16',
        'AWAY_HP_CT': 'Int16',
        'AWAY_BB_CT': 'Int16',
        'AWAY_IBB_CT': 'Int16',
        'AWAY_SO_CT': 'Int16',
        'AWAY_SB_CT': 'Int16',
        'AWAY_CS_CT': 'Int16',
        'AWAY_GDP_CT': 'Int16',
        'AWAY_XI_CT': 'Int16',
        'AWAY_PITCHER_CT': 'Int16',
        'AWAY_ER_CT': 'Int16',
        'AWAY_TER_CT': 'Int16',
        'AWAY_WP_CT': 'Int16',
        'AWAY_BK_CT': 'Int16',
        'AWAY_PO_CT': 'Int16',
        'AWAY_A_CT': 'Int16',
        'AWAY_PB_CT': 'Int16',
        'AWAY_DP_CT': 'Int16',
        'AWAY_TP_CT': 'Int16',
        'HOME_AB_CT': 'Int16',
        'HOME_2B_CT': 'Int16',
        'HOME_3B_CT': 'Int16',
        'HOME_HR_CT': 'Int16',
        'HOME_BI_CT': 'Int16',
        'HOME_SH_CT': 'Int16',
        'HOME_SF_CT': 'Int16',
        'HOME_HP_CT': 'Int16',
        'HOME_BB_CT': 'Int16',
        'HOME_IBB_CT': 'Int16',
        'HOME_SO_CT': 'Int16',
        'HOME_SB_CT': 'Int16',
        'HOME_CS_CT': 'Int16',
        'HOME_GDP_CT': 'Int16',
        'HOME_XI_CT': 'Int16',
        'HOME_PITCHER_CT': 'Int16',
        'HOME_ER_CT': 'Int16',
        'HOME_TER_CT': 'Int16',
        'HOME_WP_CT': 'Int16',
        'HOME_BK_CT': 'Int16',
        'HOME_PO_CT': 'Int16',
        'HOME_A_CT': 'Int16',
        'HOME_PB_CT': 'Int16',
        'HOME_DP_CT': 'Int16',
        'HOME_TP_CT': 'Int16',
    }

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, dtype=dtypes)
        df = df.drop(columns=df.filter(regex='.*NAME_TX$').columns)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df.columns = map(str.lower, df.columns)
    df.columns = df.columns.str.replace('_ct', '')
    df = df.rename(columns={
        'game_id': 'GAME_ID',
        'park_id': 'ParkID',
        'home_hits': 'home_h',
        'away_hits': 'away_h',
    })

    return df

def process_data(df, stable=True):
    """Process raw data into useful files for model."""
    df['date'] = pd.to_datetime(df['game_dt'], format='%Y%m%d')
    del df['game_dt']
    df['year'] = df.date.dt.year

    df['home_pa']  = (df['home_ab'] + df['home_bb'] +
        df['home_hp'] + df['home_sf'] + df['home_sh'])
    df['away_pa']  = (df['away_ab'] + df['away_bb'] +
        df['away_hp'] + df['away_sf'] + df['away_sh'])

    # df['windftob'] =

    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()
