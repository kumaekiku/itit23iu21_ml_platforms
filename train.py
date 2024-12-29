#!/usr/bin/env python
# coding: utf-8

import pickle
import time

import pandas as pd
from sklearn.naive_bayes import GaussianNB

from model_wrapper import NBWrapper

# get data
print('Obtaining data...')
path = "./data/train_data.csv"
df = pd.read_csv(path)
time.sleep(1)
print(df.head(10), "\n")

# preprocess
print('Preprocessing data...\n')
categorical = ["salary", "work_accident"]
numerical = ["satisfaction_level", "time_spend_company", "average_montly_hours"]
target = "left"

# get features and target
X = df.drop(columns=[target])
y = df[target].values
time.sleep(1)

# create model wrapper
print('Initialize and train model...')
wrapper = NBWrapper(model=GaussianNB(), target=target, num=numerical, cat=categorical)

# train model
wrapper.model.fit(X, y)
time.sleep(1)
print(wrapper, "\n")

# export model
output = 'wrap_model.bin'
with open(output, 'wb') as f_out:
    pickle.dump(wrapper, f_out)

print(f'Export model at: {output}')
