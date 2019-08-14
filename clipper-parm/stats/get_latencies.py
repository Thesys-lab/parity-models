import argparse
import numpy as np
import os

from stats import print_stats


class Batch:
    def __init__(self):
        self.decoded = False
        self.decoded_latencies = []
        self.latencies = []

    def add_query(self, latency, is_decoded):
        if is_decoded:
            self.decoded_latencies.append(latency)
            self.decoded = True
        else:
            self.latencies.append(latency)


def save_numbers(vals, outfile):
    with open(outfile, 'w') as out:
        out.write(','.join(["{:.4f}".format(v) for v in vals]))


def process_log(args):
    with open(args.e2e_file, 'r') as infile:
        lines = infile.readlines()

    batches = {}
    for line in lines:
        group_id = int(line.split("group_id=")[1].split(',')[0])
        batch_id = int(line.split("batch_id=")[1].split(',')[0])
        reconstruction = line.split("is_reconstruction=")[1].split(',')[0] == "true"
        latency = int(line.split("=")[-1])

        if batch_id not in batches:
            batches[batch_id] = Batch()

        batches[batch_id].add_query(latency, reconstruction)

    if args.batch_size > 1:
        for _, b in batches.items():
            if len(b.latencies) > 1:
                b.latencies = [b.latencies[-1]]
            if len(b.decoded_latencies) > 1:
                b.decoded_latencies = [b.decoded_latencies[-1]]

    og_latencies = []
    for _, b in batches.items():
        og_latencies.extend(b.latencies)

    og_med = np.median(og_latencies)

    latencies = []
    for _, b in batches.items():
        if b.decoded:
            if args.mode == "coded":
                latencies.extend(b.decoded_latencies)
            elif args.mode == "cheap":
                if len(b.latencies) == 0:
                    # If we don't have logs for "normal" latencies, then those
                    # batches were likely removed from the queue before
                    # processing due to a completed redundant task. In that
                    # case, we just take the decoded latencies.
                    latencies.extend(b.decoded_latencies)
                else:
                    # For cheap and replication, the "decoded" latencies are the
                    # minimum between the two latencies. We perform the max
                    # operation between the approximate result and the median
                    # latency so as not to log returning an approximate result
                    # if that result was returned too soon. This simulates
                    # having an aggressive timeout of median latency.
                    mins = [min(og, max(og_med, d)) for og, d in zip(b.latencies, b.decoded_latencies)]
                    latencies.extend(mins)
        else:
            latencies.extend(b.latencies)

    # Convert to ms
    latencies = [ x / 1000 for x in latencies ]
    print_stats(latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("e2e_file", type=str,
                        help="File with e2e latencies")
    parser.add_argument("out_prefix", type=str,
                        help="Prefix for outfiles")
    parser.add_argument("mode", type=str,
                        choices=["none", "equal", "coded", "cheap"],
                        help="One of {none, equal, replication, coded, cheap}")
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    process_log(args)
