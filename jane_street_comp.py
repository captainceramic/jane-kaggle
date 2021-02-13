#!/usr/bin/env python3
""" Train a model to predict the day trade data.

The data will be divided into chunks (~15 day chunks)

Target variable will be the weight*response (I will call it value)
We then make a call on 0/1 depending on the result.

Notes:
* Cross-validation is done by holding out a 15-day period, then finally train with it to eke out the last bit.

TB TODO: Plot out a correlation between weight and response.

"""

import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import xgboost as xgb


def load_datafile(data_filename):
    """ Load up a data file. """

    # 130 feature columns
    dtypes = {"feature_{}".format(i): np.float32 for i in range(130)}
    dtypes["ts_id"] = np.uint32
    dtypes["date"] = np.uint16
    dtypes["resp"] = np.float32
    dtypes["weight"] = np.float32

    month_data = pd.read_csv(data_filename, dtype=dtypes).sort_values(["date", "ts_id"])

    # drop the response columns
    month_data.drop(columns=["resp_1", "resp_2", "resp_3", "resp_4"],
                    inplace=True)

    # TB Ideas: standardise features. try to use the binary outputs from 'features.csv'

    month_data["value"] = month_data.weight * month_data.resp
    month_data["good_trade"] = month_data.value.apply(lambda x: 1 if x>0.0 else 0) 

    feature_cols = ["feature_{}".format(i) for i in range(130)]
    dtrain = xgb.DMatrix(month_data[feature_cols],
                         label=month_data.good_trade)

    return dtrain


# Use glob to find all the month files
all_filenames = [filename for filename in glob.glob("data/train_*.csv")]
date_numbers = [int(re.search(r"\d+", filename).group(0))
                for filename in all_filenames]
file_tables = pd.DataFrame({"month_number": date_numbers,
                            "filename": all_filenames}).sort_values("month_number")

# train a good XGBoost model for each month? Using the update thing?
first_month = load_datafile(file_tables.loc[file_tables.month_number == 1].filename.values[0])
max_month = file_tables.month_number.max()
rand_month = file_tables.loc[file_tables.month_number != 1].month_number.sample().values[0]
random_month = load_datafile(file_tables.loc[file_tables.month_number == rand_month].filename.values[0])

# Train the initial model on a single month.
learning_rate = 0.15
boost_rounds = 50

metric = "logloss"
params = {"booster": "gbtree",
          "eta": learning_rate,
          "max_depth": 5,
          "gamma": 2.5,
          "lambda": 0.2,
          "subsample": 0.6,
          "colsample_bytree": 0.6,
          "verbosity": 2,
          "eval_metric": metric,
          "objective": "binary:logistic"}

results = {}
evallist = [(first_month, "train"), (random_month, "eval")]
bst = xgb.train(params, first_month,
                num_boost_round=boost_rounds,
                evals_result=results,
                evals=evallist)

# Plot out some history
_, ax = plt.subplots(figsize=(16, 9))
ax.plot(results["train"][metric], label="train")
ax.plot(results["eval"][metric], label="eval")
plt.legend()
plt.savefig("first_month_results.png")

# Now, start nudging the model using later months. Plan is to
# train one month at a time.
del(first_month)
params["eta"] = learning_rate / 2.0
full_results = []
full_results.append(results)
for month_number in range(1, max_month+1):
    if month_number == rand_month:
        continue

    # Do some updates, train with early stopping.
    this_month = load_datafile(file_tables.loc[file_tables.month_number == month_number].filename.values[0])
    evallist = [(this_month, "train"), (random_month, "eval")]
    new_results = {}
    bst = xgb.train(params, this_month,
                    num_boost_round=boost_rounds,
                    early_stopping_rounds=5,
                    evals_result=new_results,
                    evals=evallist,
                    xgb_model=bst)
    full_results.append(new_results)
    
# Plot out the full results to see if this is actually working!
train_metric = np.concatenate([result["train"][metric] for result in full_results])
eval_metric = np.concatenate([result["eval"][metric] for result in full_results])

_, ax = plt.subplots(figsize=(16, 9))
ax.plot(train_metric, label="train")
ax.plot(eval_metric, label="eval")
plt.legend()
plt.savefig("multi_month_results.png")

# Final step: now we set '1' or any trade with a positive
# predicted value.
# NOTE: This is not the best way to do this! I just want
#       something with some value before I start looking into
#       methods that actually work!

examples = bst.predict(this_month) < 0.5
