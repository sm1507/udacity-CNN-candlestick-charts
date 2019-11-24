##  compute_labels.py
import pandas as pd
import numpy as np
import argparse
import os
import sys
import matplotlib.dates as dates

##########################
##  Global variables
##########################
# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-d', '--num_days', type=int, default=10, required=True)
parse_opt.print_help()
args = parse_opt.parse_args()

# Check for CSV directories
directory = 'data/csv'
datasets = ['training', 'testing']
##########################

# Compute End price > or < Start price and put in 0 or 1 directories
def compLabels(directory, dataset, filename, file):
  # Read each window dataset CSV file
  print("Computing Label for {}".format(filename))
  df = pd.read_csv(filename, parse_dates=True, index_col=0)
  df.fillna(0)
  df.reset_index(inplace=True)
  df['Date'] = df['Date'].map(dates.date2num)

  # for each window CSV file
  for line in range(0, len(df)):
    window = df.ix[line:line + args.num_days,:]
    start_price = 0
    end_price = 0
    label = ''

    # Calc End price > Start price then put in 1 directory
    if len(window) == args.num_days + 1:
      for idx, val in enumerate(window['Close']):
          if idx == 0:
            start_price = float(val)
          if idx == len(window) - 1:
            end_price = float(val)
      if end_price > start_price:
            f = open("{}/{}/1/{}_{}".format(directory, dataset, file, int(window['Date'].iloc[0])), "w+")
            f.write(window.to_csv())
            f.close()
      # 0 directory if End Price < Start price
      else:
            f = open("{}/{}/0/{}_{}".format(directory, dataset, file, int(window['Date'].iloc[0])), "w+")
            f.write(window.to_csv())
            f.close()

#  Create directories if not created
for dataset in datasets:
  if not os.path.isdir(directory):
    print("No CSV files to process")
  if not os.path.isdir("{}/{}".format(directory, dataset)):
    print("No Training or Testing CSV directory to process files")
  if not os.path.isdir("{}/{}/0".format(directory, dataset)):
    os.mkdir("{}/{}/0".format(directory, dataset))
  if not os.path.isdir("{}/{}/1".format(directory, dataset)):
    os.mkdir("{}/{}/1".format(directory, dataset))

# Execute compLabels for each file in data/csv/<dataset>/
for dataset in datasets:
  dir = os.listdir("{}/{}".format(directory, dataset))
  for file in dir:
    filename = "{}/{}/{}".format(directory, dataset, file)
    if os.path.isfile(filename):
      compLabels(directory, dataset, filename, file)
