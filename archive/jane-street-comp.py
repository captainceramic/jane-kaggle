#!/usr/bin/env python3

""" The basic plan:

1. Due to memory constraints, train up a model one day at a time.
2. Load up one day's data. Use online training with xgboost to update the model.
3. Validate using the final month's worth of data.

"""

import glob
import re

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import xgboost as xgb


# I am trying something weird with loss functions here as I am not sure
# how to do this! The idea is that I replace the +/- with a 1/0 loss function
# and then weight according to the weightings.
def load_day(day_number, filename_ds):
    """ Load up a CSV for a given day.

    Days run from 0.
    
    """

    example_filename = filename_ds.loc[filename_ds.date_number==day_number].filename.to_numpy()[0]
    this_ds = pd.read_csv(example_filename).sort_values(["date", "ts_id"])

    this_ds["good_trade"] = ((this_ds.resp * this_ds.weight) > 0.0).astype(np.int)
    this_ds["trade_weight"] = (this_ds.resp * this_ds.weight).abs()
    this_ds["trade_value"] = this_ds.resp * this_ds.weight
    
    return this_ds


# First up - data load.
# The files have been split into one day / file chunks.
day_files = [day_file for day_file in glob.glob("data/splitfiles/day*.csv")]
day_file_numbers = [(int(re.search("\d+", day_file).group(0)), day_file)
                    for day_file in day_files]
day_file_numbers.sort(key=lambda x: x[0])
day_file_numbers = pd.DataFrame(day_file_numbers, columns=["date_number", "filename"])

# Load the features and example testing data
features = pd.read_csv("data/features.csv")
example_test = pd.read_csv("data/example_test.csv")

# We'll do a 80% train/validation split
train_split = 400
day_file_numbers["train"] = day_file_numbers.date_number < train_split

# Do some data analysis, because this data is all pretty weird.
first_day = load_day(0, day_file_numbers)

# Look at histograms of weights and responses
_, ax = plt.subplots(figsize=(16, 9))
ax.hist(first_day.weight)
plt.savefig("weight_histogram.png")

_, ax = plt.subplots(figsize=(16, 9))
ax.hist(first_day.resp)
plt.savefig("response_histogram.png")

# Plan: train a model for the first day. Then, for every other day we just nudge it.
num_trees = 50
params = {"booster": "gbtree",
          "eta": 0.5,
          "gamma": 2.0,
          "max_depth": 6,
          "lambda": 1.1,
          "colsample_bytree": 0.75,
          "process_type": "default",
          "objective": "reg:squarederror"}

# Now, we begin a training loop. One day at a time.
# There are 130 feature columns
num_rounds = 250
early_stopping = 10
feature_cols = ["feature_{}".format(i) for i in range(130)]

# Maybe I want this multiple times?
first_day = True
for day_number in day_file_numbers.loc[day_file_numbers.train].date_number:
    print("processing day: {}".format(day_number+1))

    # Once we move to day 1, update rather than build new model.
    if not first_day:
        params["process_type"] = "update"
        params["updater"] = "refresh"
        params["refresh_leaf"] = 1
        params["eta"] = 0.1
        
    # Grab a random day to test against (this is going to get weird)
    test_day = day_file_numbers.loc[~day_file_numbers.train].date_number.sample()

    train_data = load_day(day_number, day_file_numbers)
    test_data = load_day(test_day.to_numpy()[0], day_file_numbers)
    
    dtrain = xgb.DMatrix(train_data[feature_cols],
                         label=train_data.trade_value)

    dvalid = xgb.DMatrix(test_data[feature_cols],
                         label=test_data.trade_value)
    
    evallist = [(dvalid, "eval"), (dtrain, "train")]

    if first_day:
        # TB NOTE: Don't do early stopping, I want all the trees.
        bst = xgb.train(params, dtrain, num_rounds, evallist)
    else:
        bst= xgb.train(params, dtrain, num_rounds, evallist,
                       xgb_model=bst, early_stopping_rounds=early_stopping)
    
    first_day = False
        
# Predict on the output
dtest = xgb.DMatrix(example_test[feature_cols])
example_test["results"] = bst.predict(dtest)

example_test["output"] = example_test.results > 0.0
