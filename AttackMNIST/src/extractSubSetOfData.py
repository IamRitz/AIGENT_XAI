import csv
import numpy as np

count = 0
file = "inputs"
filename = "../data/"+file+"-1.csv"
fields = []
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
        count = count + 1
        if count>=500:
            break
    print(np.shape(rows))
    with open("../data/"+file+".csv", 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)