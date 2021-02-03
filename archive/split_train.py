#!/usr/bin/env python3

import os.path
import re


datefile = "splitfiles/day_{}.csv"
with open("train.csv") as inputfile:
    firstline = inputfile.readline()

    # Now read each line, one at a time.
    line = True
    while line:
        line = inputfile.readline()

        # TB NOTE: This breaks at the end.
        date = re.search("\d+", line).group(0)

        # Write the output file.
        thisfile = datefile.format(date)
        if os.path.isfile(thisfile):
            with open(thisfile, "a") as outputfile:
                outputfile.write(line)
        else:
            print("creating file: {}".format(thisfile))
            with open(thisfile, "w") as outputfile:
                outputfile.write(firstline)
                outputfile.write(line)


        
        
        
