#!/bin/zsh

# rm -f data/raw/events/*
# wget -P data/raw/events https://www.retrosheet.org/events/{2020..1920..10}seve.zip
# unzip "data/raw/events/*seve.zip" -d data/raw/events

cd data/raw/events
for year in {1920..2020}
do
  cwevent -f 0-6,8-9,12-13,16-17,26-40,43-45,51,58-61 -x 0,11,12,20,45,51 -y ${year} ${year}*.EV* > Events${year}.txt
  cwgame -n -f 0-83 -x 0-94 -y ${year} ${year}*.EV* > Games${year}.txt
done
cd -
