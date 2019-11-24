##  build_baseline.py
import pandas as pd
import numpy as np
import argparse
import os
import sys
from stockstats import StockDataFrame as sdf
import matplotlib.dates as dates


##########################
##  Global variables
##########################
# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-f', '--csv_directory', type=str,
                       default='data/csv')
parse_opt.add_argument('-d', '--num_days', type=int, default=10)             
args = parse_opt.parse_args()

directory = 'data/baseline'
datasets = ['training', 'testing']
labels = ['0', '1']
##########################

# Compute End price > or < Start price and put in 0 or 1 directories
def compLabels(directory, dataset, file):
  print("Computing Label for {}".format(file))
  #ticker, csv = file.split(".")
  # Read each window dataset CSV file & do reset/cleanup
  file_path = "data/baseline/{}/{}".format(dataset, file)
  df = pd.read_csv(file_path, parse_dates=True, index_col=0)
  df.fillna(0)
  df.reset_index(inplace=True)
  
  # Seed ticker so we do lose track of the stock
  df['ticker'] = ''
  
  # Seed price after num_days for label calc
  df['endprice'] = float(0)
    
  # Add Closing price of stock on {num_days} day and use it to calc gain or 
  # loss on the stock, ex. closing price on the 10th day - closing price today
  # Compute label based on +/- of gainloss column
  for line in range(0, len(df) - args.num_days):
    
    df.ticker.iloc[line] = file.split('.')[0]
    df.endprice.iloc[line] = df.iloc[line + args.num_days]['close']
    df['gainloss'] = df['endprice'] - df['close']
    df['label'] = df['gainloss'].apply(lambda x: 1 if x > 0 else 0)

  # Clean up remove extra lines
  df = df[df.endprice != 0]

  # Normalize for ML, less than 1
  df['rsi_percent'] = df.rsi.div(100)

  # Delete more unwanted columns. Not needed as ML features
  del df['high']
  del df['open']
  del df['close'] 
  del df['low']
  del df['volume']
  del df['adj close']
  del df['endprice']
  del df['gainloss']
  del df['rsi']

  return df
  
def buildStockstatsDF(directory, dataset, file, fullfilepath): #, label, csvfile):
  # Read each CSV window file
  print("Building Stock Stats dataframe for {}".format(file))
  df = pd.read_csv(fullfilepath, parse_dates=True, index_col=0)
  df.fillna(0)
  df.reset_index(inplace=True)

  # Use StockStats to add RSI
  stockstats_df = sdf.retype(df)
  df['rsi'] = stockstats_df['rsi_14']
  
  # Delete unwanted columns
  del df['close_-1_s']
  del df['close_-1_d']
  del df['rs_14']
  del df['rsi_14']
  
  return df
  

### Main ###
for dataset in datasets:      
  # Create directories if not already created    
  if not os.path.isdir(directory):
    os.mkdir("{}".format(directory))
  if not os.path.isdir("{}/{}".format(directory, dataset)):
    os.mkdir("{}/{}".format(directory, dataset))

  # run stock data from yahoo CVS thru stockstats module to calc
  # RSI values    
  files = os.listdir("data/csv/{}".format(dataset))
  for file in files:
    if os.path.isfile("data/csv/{}/{}".format(dataset, file)):
      fullfilepath = "data/csv/{}/{}".format(dataset, file)
      df = buildStockstatsDF(directory, dataset, file, fullfilepath)
      f = open('{}/{}/{}.csv'.format(directory, dataset, file), 'w+')  
      f.write(df.to_csv(header=True))
      f.close()

  # Calc label based on if closing price of stock on {num_days} day is +/-
  # than the closing price today
  f = open('{}/{}_data.csv'.format(directory, dataset), 'w+')   #
  files = os.listdir("data/baseline/{}".format(dataset))
  for file in files:
    if os.path.isfile("data/baseline/{}/{}".format(dataset, file)):
      df = compLabels(directory, dataset, file) 
      f.write(df.to_csv(header=True))
            
  f.close() 