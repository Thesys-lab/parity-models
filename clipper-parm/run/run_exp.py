import argparse
import json
import os
from subprocess import call

POSSIBLE_REDUNDANCY_MODES = ["none", "equal", "coded", "cheap"]
POSSIBLE_QUEUE_MODES = ["single_queue", "rr"]
POSSIBLE_WORKER_TYPES = ["c5.xlarge", "p2.xlarge"]
POSSIBLE_MODELS = ["resnet18", "resnet152"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Path to JSON configuration file")
    parser.add_argument("role", type=str, choices=["launch", "run"],
                        help="Whether the runner is used for loaunching or running")
    parser.add_argument("--final_dir", type=str, help="Final directory name")
    parser.add_argument("--idx", type=int,
                        help="Index of copy to run. Must be set if"
                             " |role| is 'run'")
    parser.add_argument("--local", action="store_true",
                        help="Whether this should be a local launch")
    args = parser.parse_args()

    with open(args.config_file, 'r') as infile:
        cfg = json.load(infile)

    launcher = (args.role == "launch")
    if not launcher and args.idx is None:
        raise Exception("--idx must be set if `role` is 'launch'")

    if not args.final_dir:
        args.final_dir = '0'

    exp_id = cfg["experiment_id"]
    redundancy_modes = cfg["redundancy_modes"]
    for r in redundancy_modes:
        if r not in POSSIBLE_REDUNDANCY_MODES:
            raise Exception("Unrecognized redundancy mode '{}'".format(r))

    queue_modes = cfg["queue_modes"]
    for q in queue_modes:
        if q not in POSSIBLE_QUEUE_MODES:
            raise Exception("Unrecognized queue mode '{}'".format(q))

    frontend_type = cfg["frontend_type"]

    worker_type = cfg["worker_type"]
    if worker_type not in POSSIBLE_WORKER_TYPES:
        raise Exception("Unsupported worker type '{}'".format(worker_type))

    client_type = cfg["client_type"]

    model = cfg["model"]
    if model not in POSSIBLE_MODELS:
        raise Exception("Unsupported model '{}'".format(model))

    num_models = cfg["num_models"]
    ec_k_val = cfg["ec_k_val"]
    batch_sizes = cfg["batch_sizes"]
    num_clients = cfg["num_clients"]

    srs = cfg["send_rates"]
    send_rates = [r // num_clients for r in srs]

    # Number of "warmup" queries
    # Chosen to be greater than 200 and a multiple of 2, 3, and 4
    num_ignore = 204

    base_num_queries = cfg["num_queries"] * max(batch_sizes)
    num_trials = cfg["num_trials"]
    build_mode = "release"

    # Whether to reuse existing AWS instances.
    reuse = 1

    background_traffic = cfg["background_traffic"]
    if background_traffic[0] == "clipper":
        launch_bg = '1'
    else:
        launch_bg = '0'

    # Directory on EC2 instance containing the test data we use.
    img_dir = "/home/ubuntu/cat_v_dog/test1"

    # Number of images to iterate through. This is used to limit the amount of
    # processing that the sending client needs to do. By setting this to be
    # less than `base_num_queries`, the sending client can save work.
    num_img = 500

    if launcher:
        # Launch the maximum number of resources needed to run any configuration
        k_to_launch = min(ec_k_val)
        mode = redundancy_modes[0]
        queue_mode = queue_modes[0]

        # The number of redundant models is the number of "original" models
        # divided by the k factor.
        num_red_models = num_models // k_to_launch
        total_num_models = num_models + num_red_models

        if not args.local:
            # Launch EC2 instances and start running experiments from this
            # configuration. This call will block until all experiments complete.
            call(["./launch_distributed.sh", exp_id, mode, queue_mode,
                  str(k_to_launch), str(total_num_models), str(num_red_models),
                  frontend_type, worker_type, client_type, str(num_clients),
                  build_mode, str(reuse), args.final_dir, args.config_file,
                  launch_bg])
        else:
            call(["./launch_local.sh",  mode, queue_mode,
                  str(k_to_launch), str(total_num_models), str(num_red_models),
                  str(num_clients), build_mode, args.final_dir,
                  args.config_file])

    # Iterate the different configurations
    s3_path = cfg["s3_path"]
    for bg in background_traffic:
        launch_bg = bg
        for batch_size in batch_sizes:
            for k in ec_k_val:
                for queue_mode in queue_modes:
                    for rate in send_rates:
                        total_num_queries = base_num_queries + (num_ignore * batch_size)

                        # Our sending client counts requests in units of
                        # coding groups, which is a group of k queries. Note
                        # that the client will still send these queries
                        # individually. This is mostly done to simplify logic
                        # when sending "warmup" requests so that we don't split
                        # a coding group between "warmup" and "real" requests.
                        num_enc_groups = total_num_queries // k // batch_size
                        num_per_client = num_enc_groups // num_clients
                        aggregate_rate = rate * num_clients

                        for trial in range(num_trials):
                            for mode in redundancy_modes:
                                num_workers = num_models
                                if mode == "coded" or mode == "cheap":
                                    num_red_workers = num_models // k
                                elif mode == "equal":
                                    extra_workers = num_models // k
                                    num_workers = num_models + extra_workers
                                    num_red_workers = 0
                                elif mode == "none":
                                    num_red_workers = 0

                                suffix_dir = '/'.join([
                                    "wt_{}".format(worker_type),
                                    "bm_{}".format(model),
                                    "bg_{}".format(bg),
                                    "bs_{}".format(batch_size),
                                    "rate_{}".format(aggregate_rate),
                                    "k_{}".format(k),
                                    "nq_{}".format(total_num_queries),
                                    "mode_{}".format(mode),
                                    "it_{}".format(trial),
                                    args.final_dir
                                    ])
                                out_dir = '/'.join([
                                        exp_id,
                                        suffix_dir
                                    ])

                                s3_dir = s3_path + out_dir + '/0'
                                print(os.path.join(mode, str(trial)), s3_dir)

                                # If we're executing this on the frontend,
                                # start the experiment.
                                if args.local:
                                    start_script = "./start_local_experiment.sh"
                                else:
                                    start_script = "./start_experiment_from_frontend.sh"
                                if not launcher:
                                    call([start_script,
                                          out_dir,
                                          mode,
                                          queue_mode,
                                          str(num_workers),
                                          str(num_red_workers),
                                          str(k),
                                          str(batch_size),
                                          model,
                                          str(num_per_client),
                                          str(rate),
                                          str(rate),
                                          img_dir,
                                          str(num_img),
                                          build_mode,
                                          str(launch_bg),
                                          str(num_ignore)])
                                else:
                                    if args.local:
                                        local = '1'
                                    else:
                                        local = '0'
                                    call(["./get_results.sh",
                                          s3_dir, local])
