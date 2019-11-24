# get_stockdata.py
import argparse
#import pandas as pd
#import datetime
from pandas_datareader import data, wb
import os
import time

##########################
##  Global variables
##########################
# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-s', '--start_date', type=str, default='2016-01-01',
                       required=True)
parse_opt.add_argument('-e', '--end_date', type=str, default='2016-01-01',
                       required=True)
parse_opt.add_argument('-t', '--ticker', type=str, required=True)
parse_opt.add_argument('-d', '--dataset', type=str, required=True)
args = parse_opt.parse_args()
##########################

# get stock data from Yahoo API vi pandas_datareader
def get_yahoo_data(ticker, dataset, start_date, end_date):

  # Blow away old data and recreate directories
  directory="data/csv"
  if not os.path.isdir(directory):
      parent = directory.split("/")
      os.mkdir(parent[0])
      os.mkdir(directory)
  if not os.path.exists('{}/{}'.format(directory, dataset)):
      os.mkdir('{}/{}'.format(directory, dataset))

  for attempt in range(3):
    time.sleep(2)
    try:
      dat = data.get_data_yahoo(ticker, start=start_date, end=end_date)
      dat.to_csv("{}/{}/{}".format(directory, dataset, ticker))

    except Exception as e:
      if attempt < 2:
        print('Attempt {}: {}'.format(attempt+1, str(e)))
      else:
        raise
    else:
      break

# Main
get_yahoo_data(args.ticker, args.dataset, args.start_date, args.end_date)
