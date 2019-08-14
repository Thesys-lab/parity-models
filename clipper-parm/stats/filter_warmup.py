import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Grepped input file")
    parser.add_argument("num_clients", type=int,
                        help="Number of clients in experiment")
    parser.add_argument("ec_k_val", type=int,
                        help="k value used in coding")
    parser.add_argument("num_warmup", type=int,
                        help="The number of groups of k batches sent as warmup")
    parser.add_argument("num_batches_ignore", type=int,
                        help="The number of batches to ignore after the warmup period")
    parser.add_argument("redundancy_mode", type=str,
                        help="One of {none, equal, coded, replication, cheap}")
    args = parser.parse_args()

    if args.redundancy_mode in ["none", "equal"]:
        num_batches_per_group = args.ec_k_val
        mult_factor = 1.
    elif args.redundancy_mode in ["replication", "cheap"]:
        num_batches_per_group = 2 * args.ec_k_val
        mult_factor = 2.
    elif args.redundancy_mode == "coded":
        num_batches_per_group = args.ec_k_val + 1
        mult_factor = num_batches_per_group / args.ec_k_val
    else:
        assert False, "Redundancy mode '{}' not recognized".format(args.redundancy_mode)

    num_warmup_batches = args.num_clients * args.num_warmup * num_batches_per_group
    num_ignore_batches = int(args.num_batches_ignore * mult_factor)
    num_filter_batches = num_warmup_batches + num_ignore_batches
    num_filter_groups = num_filter_batches / num_batches_per_group
    with open(args.infile, 'r') as infile:
        lines = infile.readlines()

    for line in lines:
        if "batch_id" in line:
            batch_id = int(line.split("batch_id=")[-1].split(",")[0])
            if batch_id >= num_filter_batches:
                # Remove newline
                print(line[:-1])
        else:
            # Not all metrics have a batch_id attached to them.
            # NOTE: We might be able to just use group_id for this, but I'm not
            # 100% certain.
            group_id = int(line.split("group_id=")[-1].split(",")[0])
            if group_id >= num_filter_groups:
                # Remove newline
                print(line[:-1])

