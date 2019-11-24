# prep_wapper.py
import pandas as pd
import sys
import os
import subprocess
import argparse

##########################
##  Global variables
##########################
# Start/End training and testing dates
train_start = '2016-01-01'
train_end = '2017-12-31'
test_start = '2018-01-01'
test_end = '2018-12-31'

# Size of image - 50 x 50 pixels
img_size = 50

# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-f', '--csvfile', type=str, required=True)
parse_opt.add_argument('-d', '--num_days', type=int, default=10)
parse_opt.add_argument('-m', '--mode', type=str,
                       help='all|getdata|computelabels|buildimages',
                       default='all')
#parse_opt.print_help()
args = parse_opt.parse_args()
#########################

# Read stock ticker CSV files
data = pd.read_csv(args.csvfile)

# Mode GetData from Yahoo
if args.mode == 'all' or args.mode == 'getdata':
  # Cycle thru all ticker symbols
  for ticker in data['ticker']:
    try:
      # Get Training data
      print("Getting Training data from Yahoo for {}".format(ticker))
      subprocess.call( 'python get_stockdata.py -d training -t {} -s {} -e {}'.format(
            ticker, train_start, train_end), shell=True)
      # Get Testing data
      print("Getting Testing data from Yahoo for {}".format(ticker))
      subprocess.call( 'python get_stockdata.py -d testing -t {} -s {} -e {}'.format(
            ticker, test_start, test_end), shell=True)

    except Exception as identifier:
      print(identifier)

# Compute labels and put each 10 day window in a seperate CSV file
if args.mode == 'all' or args.mode == 'computelabels':
  # Compute labels & 10 day window in a seperate CSV
  try:
      # Compute labels
      print("Compute Training and Testing Labels")
      subprocess.call( 'python compute_labels.py -d {}'.format(
            args.num_days), shell=True)

  except Exception as identifier:
      print(identifier)

# Mode GetData from Yahoo
if args.mode == 'all' or args.mode == 'buildimages':
  try:
      # Get Training data
      print("Building PNG images from CSV directory of files")
      subprocess.call( 'python build_images.py -i {}'.format(
            img_size), shell=True)

  except Exception as identifier:
      print(identifier)

# Mode Compute baseline data
if args.mode == 'all' or args.mode == 'baseline':
  try:
      # Compute and build baseline dataset
      print("Compute and build baseline dataset")
      subprocess.call( 'python build_baseline.py', shell=True)

  except Exception as identifier:
      print(identifier)

