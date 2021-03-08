#!/bin/zsh

rm -f data/raw/events/*
wget -P data/raw/events https://www.retrosheet.org/events/{2020..1920..10}seve.zip
unzip "data/raw/events/*seve.zip" -d data/raw/events

cd data/raw/events
for year in {1920..2020}
do
  cwevent -y $year ${year}*.EV* > Events${year}.txt
done
cd -
