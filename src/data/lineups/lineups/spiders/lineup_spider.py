import scrapy
from pathlib import Path
import click
from scrapy.crawler import CrawlerProcess


class LineupSpider(scrapy.Spider):
    name = "lineup"

    def __init__(self, *args, **kwargs):
        super(LineupSpider, self).__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]

    def parse(self, response):
        for matchup in response.css("div.starting-lineups__matchup"):

            away_team = matchup.css(
                "span.starting-lineups__team-name--away")[0]
            away_team_name = away_team.css(
                'a.starting-lineups__team-name--link::text').get()
            away_team_code = away_team.css(
                'a.starting-lineups__team-name--link').attrib['data-tri-code']
            away_pitcher = matchup.css(
                'a.starting-lineups__pitcher--link::text')[2].get()
            away_players = matchup.css(
                'a.starting-lineups__player--link::text')[0:9].getall()

            home_team = matchup.css(
                "span.starting-lineups__team-name--home")[0]
            home_team_name = home_team.css(
                'a.starting-lineups__team-name--link::text').get()
            home_team_code = home_team.css(
                'a.starting-lineups__team-name--link').attrib['data-tri-code']
            home_pitcher = matchup.css(
                'a.starting-lineups__pitcher--link::text')[5].get()
            home_players = matchup.css(
                'a.starting-lineups__player--link::text')[9:18].getall()

            yield {
                'away_team_name': away_team_name.strip(),
                'away_team_code': away_team_code,
                'away_pitcher': away_pitcher,
                'away_player1': away_players[0],
                'away_player2': away_players[1],
                'away_player3': away_players[2],
                'away_player4': away_players[3],
                'away_player5': away_players[4],
                'away_player6': away_players[5],
                'away_player7': away_players[6],
                'away_player8': away_players[7],
                'away_player9': away_players[8],
                'home_team_name': home_team_name.strip(),
                'home_team_code': home_team_code,
                'home_pitcher': home_pitcher,
                'home_player1': home_players[0],
                'home_player2': home_players[1],
                'home_player3': home_players[2],
                'home_player4': home_players[3],
                'home_player5': home_players[4],
                'home_player6': home_players[5],
                'home_player7': home_players[6],
                'home_player8': home_players[7],
                'home_player9': home_players[8],
            }


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('-d', '--date')
def main(input_filepath, date):
    url = 'https://www.mlb.com/starting-lineups/' + date

    name = 'lineups' + date + '.csv'
    lineups_csv = Path(input_filepath) / 'Lineups' / name

    print(lineups_csv)
    process = CrawlerProcess({
        'FEEDS': {
            lineups_csv: {
                'format': 'csv',
                'overwrite': True,
            }
        }
    })

    process.crawl(LineupSpider, start_url=url)
    process.start()


if __name__ == '__main__':

    main()
