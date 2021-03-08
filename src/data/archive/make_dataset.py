# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.main_preprocessing import (
    GameLogProcessor, EventsProcessor, RostersProcessor)
from src.data.build_panel import PanelBuilder

# from src.data.lineup_preprocessing import LineupProcessor


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # lineup_process = LineupProcessor()
    # lineup_process.scrape_data(input_filepath)
    # lineup_process.read_data(input_filepath)
    # lineup_process.process_data(output_filepath)
    # lineup_process.write_data(Path(output_filepath) / 'lineup.pkl')

    game_log_process = GameLogProcessor()
    game_log_process.read_data(input_filepath)
    game_log_process.process_data()
    game_log_process.write_data(Path(output_filepath) / 'game_log.pkl')

    event_process = EventsProcessor()
    event_process.read_data(input_filepath)
    event_process.process_data()
    event_process.write_data(output_filepath)

    rosters_process = RostersProcessor()
    rosters_process.read_data(input_filepath)
    rosters_process.process_data()
    rosters_process.write_data(Path(output_filepath) / 'rosters.pkl')

    panel_builder = PanelBuilder()
    panel_builder.build_panel(output_filepath)
    panel_builder.write_data(Path(output_filepath) / 'panel.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
