##  run_baseline.py
import pandas as pd
import numpy as np
from time import time
import os
import sys
import datetime
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# Train & predict ML
def train_predict(learner):

  # Load data files
  df_train = pd.read_csv("data/baseline/training_data.csv", parse_dates=True, index_col=0)
  df_test = pd.read_csv("data/baseline/testing_data.csv", parse_dates=True, index_col=0)
  df_train.reset_index(inplace=True)
  df_test.reset_index(inplace=True)
    
  # Convert from string to numeric...  python auto thinks they are string  
  df_train = df_train.convert_objects(convert_numeric=True)
  df_test = df_test.convert_objects(convert_numeric=True)
  
  # Set training & test arrays.  Need to reshape since there is only one feature
  X_train = df_train['rsi_percent'].values.reshape(-1,1)
  y_train = df_train['label']
  X_test = df_test['rsi_percent'].values.reshape(-1,1)
  y_test = df_test['label']

  # Change NAN (no data) data to 0.  NAN was causing miscaluations & errors
  X_train[np.isnan(X_train)]=0
  y_train[np.isnan(y_train)]=0
  X_test[np.isnan(X_test)]=0
  y_test[np.isnan(y_test)]=0
  
  # Learn fit the traing data
  learner.fit(X_train, y_train)
  
  # Set predictions to array
  predictions_train = learner.predict(X_train)
  predictions_test = learner.predict(X_test)

  # Result output array
  results = {}
  results['learner'] = learner.__class__.__name__
  results['acc_train'] = accuracy_score(y_train, predictions_train)
  results['acc_test'] = accuracy_score(y_test, predictions_test)
  results['f_train'] = fbeta_score(y_train, predictions_train, beta = .5)
  results['f_test'] = fbeta_score(y_test, predictions_test, beta = .5)

  # Calc recall, precision from confusion matrix
  TN, FP, FN, TP = confusion_matrix(y_test, predictions_test).ravel()
  results['recall'] = TP / (TP+FN)
  results['precision'] = TP / (TP+FP)

  f = open("./results.txt", 'a')
  date = datetime.datetime.now()
  f.write("\n\n#### {} ####".format(date))
  for key,val in results.items():
        f.write ("\n{}:  {}".format(key, val))
        print("\n{}:  {}".format(key, val))
  f.close()
  print("{} trained on {} samples.".format(learner.__class__.__name__, "samplesszie"))

  print(results)
  print(TN, FP, FN, TP)

############# Main ###############
# Ran several different algoritms 
learner1 = LogisticRegression(solver='liblinear', random_state=42) 
learner2 = GradientBoostingClassifier(random_state=42)
learner3 = GaussianNB()

for learner in [learner1, learner2, learner3]:
  train_predict(learner)