##  build_images.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
from mpl_finance import candlestick2_ochl
import sys


##########################
##  Global variables
##########################
# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-i', '--image_size', type=int, default=50, required=True)
parse_opt.print_help()
args = parse_opt.parse_args()

directory = 'data/images'
datasets = ['training', 'testing']
labels = ['0', '1']

# Chart Image sizes
dpi = 96
dimension = args.image_size
##########################

# Create PNG image from the CSV window
def buildImage(directory, dataset, filename, label, csvfile):
  # Read each CSV window file
  print("Building image for {}".format(filename))
  #print("directory = {}, dataset = {}, filename = {} , label = {}, csvfile = {} ".format(directory, dataset, filename, label, csvfile))
  fname = filename.split('/')[4]
  df = pd.read_csv(filename, parse_dates=True, index_col=0)
  df.fillna(0)
  df.reset_index(inplace=True)

  # Plot Chart
  plt.style.use('dark_background')
  fig = plt.figure(figsize = (dimension / dpi, dimension / dpi ), dpi = dpi)
  chart = fig.add_subplot(1,1,1)
  candlestick2_ochl(chart, df['Open'], df['Close'], df['High'], df['Low'],
                      width=1, colorup='#77d879', colordown='#db3f3f')
  chart.grid(False)
  chart.set_xticklabels([])
  chart.set_yticklabels([])
  chart.yaxis.set_visible(False)
  chart.xaxis.set_visible(False)
  chart.axis('off')

  # Save chart image
  imgfile = '{}/{}/{}/{}.png'.format(directory, dataset, label, fname)
  print("Image File: {}".format(imgfile))
  fig.savefig(imgfile, pad_inches=0, transparent=False)
  plt.close(fig)

#  Create directories if not yet created
for dataset in datasets:
  if not os.path.isdir(directory):
    os.mkdir("{}".format(directory))
  if not os.path.isdir("{}/{}".format(directory, dataset)):
    os.mkdir("{}/{}".format(directory, dataset))
  if not os.path.isdir("{}/{}/0".format(directory, dataset)):
    os.mkdir("{}/{}/0".format(directory, dataset))
  if not os.path.isdir("{}/{}/1".format(directory, dataset)):
    os.mkdir("{}/{}/1".format(directory, dataset))

# List all files in CSV directories to execute against buildImage funct
for dataset in datasets:
  for label in labels:
    dir = os.listdir("data/csv/{}/{}".format(dataset, label))
    for csvfile in dir:
      filename = "data/csv/{}/{}/{}".format(dataset, label, csvfile)
      if os.path.isfile(filename):
        buildImage(directory, dataset, filename, label, csvfile)
