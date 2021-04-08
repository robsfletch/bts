import scrapy
from pathlib import Path
import click
from scrapy.crawler import CrawlerProcess


class RotowireSpider(scrapy.Spider):
    name = "rotowire"

    def __init__(self, *args, **kwargs):
        super(RotowireSpider, self).__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]

    def parse(self, response):
        for matchup in response.xpath('//div[(contains(@class,"lineup") and contains(@class,"is-mlb")) and not(contains(@class,"is-tools")) and not(contains(@class, "is-postponed"))]'):
            game_time = matchup.css('div.lineup__time::text').get()
            away_team = matchup.css('div[class="lineup__team is-visit"]')
            away_team_code = away_team.css('div.lineup__abbr::text').get()

            away_team_name = matchup.css(
                'div[class="lineup__mteam is-visit"]::text'
            ).get().strip()

            away_lineup = matchup.css('ul[class="lineup__list is-visit"]')
            away_pitcher = away_lineup.css(
                'div[class="lineup__player-highlight-name"]'
            ).css('a::text').get()

            away_players = away_lineup.css('li[class="lineup__player"]')

            home_team = matchup.css('div[class="lineup__team is-home"]')
            home_team_code = home_team.css('div.lineup__abbr::text').get()

            home_team_name = matchup.css(
                'div[class="lineup__mteam is-home"]::text'
            ).get().strip()

            home_lineup = matchup.css('ul[class="lineup__list is-home"]')
            home_pitcher = home_lineup.css(
                'div[class="lineup__player-highlight-name"]'
            ).css('a::text').get()

            home_players = home_lineup.css('li[class="lineup__player"]')

            yield {
                'game_time': game_time,
                'away_team_name': away_team_name,
                'away_team_code': away_team_code,
                'away_pitcher': away_pitcher,
                'away_player1': away_players[0].css('a').attrib['title'],
                'away_player2': away_players[1].css('a').attrib['title'],
                'away_player3': away_players[2].css('a').attrib['title'],
                'away_player4': away_players[3].css('a').attrib['title'],
                'away_player5': away_players[4].css('a').attrib['title'],
                'away_player6': away_players[5].css('a').attrib['title'],
                'away_player7': away_players[6].css('a').attrib['title'],
                'away_player8': away_players[7].css('a').attrib['title'],
                'away_player9': away_players[8].css('a').attrib['title'],
                'home_team_name': home_team_name,
                'home_team_code': home_team_code,
                'home_pitcher': home_pitcher,
                'home_player1': home_players[0].css('a').attrib['title'],
                'home_player2': home_players[1].css('a').attrib['title'],
                'home_player3': home_players[2].css('a').attrib['title'],
                'home_player4': home_players[3].css('a').attrib['title'],
                'home_player5': home_players[4].css('a').attrib['title'],
                'home_player6': home_players[5].css('a').attrib['title'],
                'home_player7': home_players[6].css('a').attrib['title'],
                'home_player8': home_players[7].css('a').attrib['title'],
                'home_player9': home_players[8].css('a').attrib['title'],
            }

@click.command()

@click.argument('input_filepath', type=click.Path(exists=True))
@click.option('-d', '--date')
@click.option('-t', '--tomorrow')
def main(input_filepath, date, tomorrow):
    if tomorrow == 'True':
        url = 'https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow'
    else:
        url = 'https://www.rotowire.com/baseball/daily-lineups.php'

    print(url)
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

    process.crawl(RotowireSpider, start_url=url)
    process.start()


if __name__ == '__main__':

    main()
