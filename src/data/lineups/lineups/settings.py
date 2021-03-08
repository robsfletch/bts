# Scrapy settings for lineups project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'lineups'

SPIDER_MODULES = ['lineups.spiders']
NEWSPIDER_MODULE = 'lineups.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True
