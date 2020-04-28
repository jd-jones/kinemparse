import statistics
import os
import csv
import argparse

import yaml

from mathtools import utils


def main(out_dir=None, results_file=None):
    out_dir = os.path.expanduser(out_dir)
    results_file = os.path.expanduser(results_file)

    outputMean = os.path.join(out_dir, 'output_mean.csv')
    outputSTD = os.path.join(out_dir, 'output_std.csv')

    data = []

    # mean = []
    # std = []

    kernel_sizes = []
    loss = []
    acc = []
    prc = []
    rec = []
    f1 = []
    metrics = [loss, acc, prc, rec, f1]

    with open(results_file) as csvDataFile:
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
                data[i][j] = float(data[i][j][:-1]) / 100

            # loss.append(float(row[0]))
            # acc.append(float(row[1][:-1])/100)
            # prc.append(float(row[2][:-1])/100)
            # rec.append(float(row[3][:-1])/100)
            # f1.append(float(row[4][:-1])/100)

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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--results_file')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    if os.path.exists(config_file_path):
        with open(config_file_path, 'rt') as config_file:
            config = yaml.safe_load(config_file)
    else:
        config = {}

    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
