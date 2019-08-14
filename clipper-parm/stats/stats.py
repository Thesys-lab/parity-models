import argparse
import numpy as np


def print_stats(latencies):
    if len(latencies) == 0:
        print("none")
        return

    num = len(latencies)
    med = np.median(latencies)
    mean = np.mean(latencies)
    p99 = np.percentile(latencies, 99)
    p995 = np.percentile(latencies, 99.5)
    p999 = np.percentile(latencies, 99.9)

    delim = '\t'
    print(delim.join(["NUM", "MED", "MEAN", "P99", "P99.5", "P99.9"]))
    print(delim.join([str(num)] + ["{:.2f}".format(x) for x in [med, mean, p99, p995, p999]]))


def get_stats(args):
    with open(args.infile, 'r') as infile:
        lines = infile.readlines()

    latencies = []
    for line in lines:
        latencies.append(int(line.split('=')[-1]))

    latencies = [l / args.div_factor for l in latencies]
    print_stats(latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str,
                        help="Path to file containing stats")
    parser.add_argument("--div_factor", type=float, default=1.,
                        help="Amount to dividie stats values by")
    args = parser.parse_args()

    get_stats(args)
