#!/usr/bin/env python3

""" We have a problem - the data is too big to fit in memory!

Instead we will save it out one period at a time

"""

import re
import os.path


SPLIT_PERIOD = 15
INPUT_DATA = "train.csv"
OUTPUT_PATTERN = "train_{}.csv"

with open(INPUT_DATA, "r") as inputfile:
    headerline = inputfile.readline()
    line = True
    while line:
        line = inputfile.readline()
        if line:
            # Process a given line.
            date_number = int(re.search("\d+", line).group(0))
            month_number = (date_number // SPLIT_PERIOD) + 1
            month_path = OUTPUT_PATTERN.format(month_number)

            if os.path.exists(month_path):
                with open(month_path, "a") as month_file:
                    month_file.write(line)
            else:
                with open(month_path, "w") as month_file:
                    month_file.write(headerline)
                    month_file.write(line)

