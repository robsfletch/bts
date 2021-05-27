from pybaseball import batting_stats_range
from pybaseball import cache
cache.enable()
cache.config.cache_type='csv'
cache.config.save()

import datetime
today = datetime.datetime.today().strftime('%Y-%m-%d')
today

opener = datetime.date(2021, 4, 1)
opener

for day in range(0, )
data = batting_stats_range('2021-04-01', '2021-04-01')
data
