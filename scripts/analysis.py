
import statistics
import os
import sys


output_dir = '~/repo/kinemparse/data/output/predict-activity'
inputPath = os.path.join(output_dir, 'results.csv')
inputPath = os.path.expanduser(inputPath)

outputMean = os.path.join(output_dir, 'output_mean.csv')
outputMean = os.path.expanduser(outputMean)

outputSTD = os.path.join(output_dir, 'output_std.csv')
outputSTD = os.path.expanduser(outputSTD)

import csv

data = []

mean = []
std = []

kernel_sizes = []
loss = []
acc = []
prc = []
rec = []
f1 = []
metrics = [loss, acc, prc, rec, f1]

with open(inputPath) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        data.append(row)


for i in range(len(data)):
    for j in range(len(data[0])):
        if j == 0:
            data[i][j] = int(data[i][j])
            if data[i][j] not in kernel_sizes:
                kernel_sizes.append(data[i][j])
        if j == 1:
            data[i][j] = float(data[i][j])
        if j > 1:
            data[i][j] = float(data[i][j][:-1])/100

        #loss.append(float(row[0]))
        #acc.append(float(row[1][:-1])/100)
        #prc.append(float(row[2][:-1])/100)
        #rec.append(float(row[3][:-1])/100)
        #f1.append(float(row[4][:-1])/100)

summary_mean = []
summary_std = []


for kernel_size in kernel_sizes:
    loss = []
    acc = []
    prc = []
    rec = []
    f1 = []
    metrics = [loss, acc, prc, rec, f1]
    kernel_mean = [kernel_size]
    kernel_std = [kernel_size]
    print()
    print("trial")
    for trial in data:
        if trial[0] == kernel_size:
            print(trial)
            loss.append(trial[1])
            acc.append(trial[2])
            prc.append(trial[3])
            rec.append(trial[4])
            f1.append(trial[5])
    print("kernel size: ")
    print(kernel_size)
    print("loss")
    print(loss)
    for metric in metrics:
        kernel_mean.append(sum(metric) / len(metric))
        kernel_std.append(statistics.stdev(metric))
    summary_mean.append(kernel_mean)
    summary_std.append(kernel_std)

with open(outputMean,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(summary_mean)

with open(outputSTD,"w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(summary_std)
