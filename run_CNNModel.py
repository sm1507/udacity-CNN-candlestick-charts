##  run_CNNModel.py
import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import tensorflow as tf
from sklearn.datasets import load_files
from keras.utils import np_utils
#from glob import glob
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import backend as K
from contextlib import redirect_stdout
##########################
##  Global variables
##########################
# GPU Settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Parse option variables from script executable
parse_opt = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse_opt.add_argument('-i', '--image_size', type=int, default=50)
args = parse_opt.parse_args()

directory = 'data/images'
datasets = ['training', 'testing']
labels = ['0', '1']

batch_size = 64
epoch = 10
#########################

def load_dataset(directory):
  data = load_files(directory)
  stock_img_files = np.array(data['filenames'])
  stock_img_targets = np_utils.to_categorical(np.array(data['target']), 2)
  return stock_img_files, stock_img_targets

# Note: preprocessing code from Udacity dog images project
def path_to_tensor(img_path):
  # Default" dimension = 50 x 50px
  dimension = args.image_size
  img = image.load_img(img_path, target_size=(dimension, dimension))
  x = image.img_to_array(img)
  return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
  list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
  return np.vstack(list_of_tensors)



def buildModel(dimension):
  # Build layers, initial layers were obtained from 'Using Deep Learning Neural
  # Networks and Candlestick Chart Representation to Predict Stock Market' by
  # Dept of Computer Science & Engineering Taiwan, Rosdyanna Mangir Irawan Kusuma
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=3,strides=3, padding='same',
                 activation='relu', input_shape = (dimension, dimension, 3)))
  model.add(Conv2D(filters=48, kernel_size=3,strides=3, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(filters=64, kernel_size=3,strides=3, padding='same',
                 activation='relu'))
  model.add(Conv2D(filters=96, kernel_size=3,strides=3, padding='same',
                  activation='relu'))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(output_dim=256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(output_dim=2, activation='softmax'))
  return model

##########################
##  Main
##########################


# Load data sets
train_img_files, train_img_targets = load_dataset("{}/training".format(directory))
test_img_files, test_img_targets = load_dataset("{}/testing".format(directory))

# Open file to Capture notes in Summary file
f = open("Summary_Report.txt", "w+")
f.write("\n====== {} ======\n".format(datetime.now()))

# Take some prelimnary data notes
f.write("\nTotal number of images {}".format(len(np.hstack([train_img_files, test_img_files]))))
f.write("\nTotal Train images {}".format(len(train_img_files)))
f.write("\nTotal Test images {}".format(len(test_img_files)))

for dataset in datasets:
  for label in labels:
    f.write("\nTotal num images for dataset {}, Label {} is {}".format(
             dataset, label, len(os.listdir("{}/{}/{}".format(
             directory, dataset, label)))))
f.write("\n")
# Note: preprocessing from Udacity Dog Images project
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process to resize to 4 channels for tensorflow
train_tensors = paths_to_tensor(train_img_files).astype('float32')/255
test_tensors = paths_to_tensor(test_img_files).astype('float32')/255

# Get model
model = buildModel(args.image_size)

# Write Summary() to file
with redirect_stdout(f):
  model.summary()

# compile & fit
model.compile(optimizer=Adam(lr=1.0e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save weights for all epochs
checkpointer = ModelCheckpoint(filepath='weights.hdf5',
                               verbose=1, save_best_only=False)

# Fit Model                                
model.fit(train_tensors, train_img_targets, batch_size=batch_size, epochs=epoch,
          callbacks=[checkpointer], verbose=1)

# Load weights from saved checkpoint 
model.load_weights('weights.hdf5')

# Calc Accuracy
predict = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
accuracy = 100*np.sum(np.array(predict)==np.argmax(test_img_targets, axis=1))/len(predict)

# Calc Recall & Precision
predictions = model.predict(test_tensors)
y_pred = np.argmax(predictions, axis=1)
y_test = np.argmax(test_img_targets, axis=1)
TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
recall = TP / (TP+FN)
precision = TP / (TP+FP)

# Write to file
f.write('\nTest accuracy: %.4f%%' % accuracy)
f.write('\nRecall: {} \nPrecision: {}\n'.format(recall, precision))





f.close()
