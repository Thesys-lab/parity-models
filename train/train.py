import argparse
import json
import os
from shutil import copyfile

from parity_model_trainer import ParityModelTrainer


def get_config(num_epoch, ec_k, loss, encoder, decoder, base_model_file,
               base, dataset, save_dir, base_model_input_size, parity_model,
               only_test, loss_from_true_labels, cfg):
    if ec_k == 2:
        mb_size = 64
    else:
        mb_size = 32
    return {
        "save_dir": save_dir,
        "final_epoch": num_epoch,
        "ec_k": ec_k,
        "ec_r": 1,
        "batch_size": mb_size,
        "only_test": only_test,

        "Loss": loss,

        "Optimizer": {
            "class": "torch.optim.Adam",
            "args": {
                "lr": 1e-03,
                "weight_decay": 1e-05
            }
        },

        "Encoder": { "class": encoder },
        "Decoder": { "class": decoder },

        "base_model_file": base_model_file,
        "base_model_input_size": base_model_input_size,
        "BaseModel": base,

        "ParityModel": parity_model,
        "Dataset": dataset,

        "loss_from_true_labels": loss_from_true_labels,
        "train_encoder": cfg["train_encoder"],
        "train_decoder": cfg["train_decoder"],
        "train_parity_model": cfg["train_parity_model"],
    }


def get_loss(loss_type, cfg):
    from_true_labels = False
    if "KLDivLoss" in loss_type or "CrossEntropy" in loss_type:
        if not cfg["train_encoder"] or not cfg["train_decoder"]:
            raise Exception(
                    "{} currently only supported for learned encoders and decoders".format(loss_type))

    if "CrossEntropy" in loss_type:
        from_true_labels = True

    return {"class": loss_type}, from_true_labels


def get_base_model(dataset, base_model_type):
    base_path = "base_model_trained_files"

    model_file = os.path.join(
        base_path, dataset, base_model_type, "model.t7")

    num_classes = 10
    input_size = None

    if base_model_type == "base-mlp":
        base = {
            "class": "base_models.base_mlp.BaseMLP"
        }
        input_size = [-1, 784]
    elif base_model_type == "resnet18":
        if dataset == "cifar10":
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cifar100":
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True,
                    "num_classes": 100
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cat_v_dog":
            base = {
                "class": "torchvision.models.resnet18",
                "args": {
                    "pretrained": False,
                    "num_classes": 2
                }
            }
            input_size = [-1, 3, 224, 224]
        else:
            base = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": False
                }
            }
            input_size = [-1, 1, 28, 28]
    elif base_model_type == "resnet152":
        assert dataset == "cifar100", "ResNet152 only used for CIFAR-100"
        base = {
            "class": "base_models.resnet.ResNet152",
            "args": {
                "size_for_cifar": True,
                "num_classes": 100
            }
        }
        input_size = [-1, 3, 32, 32]
    elif base_model_type == "vgg11":
        assert dataset == "gcommands", "VGG currently only used for GCommands"
        base = {
            "class": "base_models.vgg.VGG11",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    elif base_model_type == "lenet":
        assert dataset == "gcommands", "LeNet currently only used for GCommands"
        base = {
            "class": "base_models.lenet.LeNet",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]

    else:
        raise Exception("Invalid base_model_type: {}".format(base_model_type))
    if dataset == "mnist":
        ds = {
            "class": "datasets.code_dataset.MNISTCodeDataset",
        }
    elif dataset == "fashion-mnist":
        ds = {
            "class": "datasets.code_dataset.FashionMNISTCodeDataset",
        }
    elif dataset == "cifar10":
        ds = {
            "class": "datasets.code_dataset.CIFAR10CodeDataset",
        }
    elif dataset == "cifar100":
        ds = {
            "class": "datasets.code_dataset.CIFAR100CodeDataset",
        }
    elif dataset == "cat_v_dog":
        ds = {
            "class": "datasets.code_dataset.CatDogCodeDataset",
        }
    elif dataset == "gcommands":
        ds = {
            "class": "datasets.gcommands_dataset.GCommandsCodeDataset",
        }
    else:
        raise Exception("Unrecognized dataset name '{}'".format(dataset))
    return model_file, base, input_size, ds


def get_parity_model(dataset, parity_model_type):
    if parity_model_type == "base-mlp":
        parity_model = {
            "class": "base_models.base_mlp.BaseMLP"
        }
        input_size = [-1, 784]
    elif parity_model_type == "resnet18":
        if dataset == "cifar10":
            parity_model = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": True
                }
            }
            input_size = [-1, 3, 32, 32]
        elif dataset == "cat_v_dog":
            parity_model = {
                "class": "torchvision.models.resnet18",
                "args": {
                    "pretrained": False,
                    "num_classes": 2
                }
            }
            input_size = [-1, 3, 224, 224]
        else:
            parity_model = {
                "class": "base_models.resnet.ResNet18",
                "args": {
                    "size_for_cifar": False
                }
            }
            input_size = [-1, 1, 28, 28]
    elif parity_model_type == "resnet152":
        assert dataset == "cifar100", "ResNet152 only used for CIFAR-100"
        parity_model = {
            "class": "base_models.resnet.ResNet152",
            "args": {
                "size_for_cifar": True,
                "num_classes": 100
            }
        }
        input_size = [-1, 3, 32, 32]
    elif parity_model_type == "vgg11":
        assert dataset == "gcommands", "VGG currently only used for GCommand"
        parity_model = {
            "class": "base_models.vgg.VGG11",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    elif parity_model_type == "lenet":
        assert dataset == "gcommands", "LeNet currently only used for GCommands"
        parity_model = {
            "class": "base_models.lenet.LeNet",
            "args": {
                "num_classes": 30
            }
        }
        input_size = [-1, 1, 161, 101]
    else:
        raise Exception("Unrecognized parity_model_type '{}'".format(parity_model_type))
    return parity_model, input_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="JSON file containing configuration parameters")
    parser.add_argument("overall_save_dir", type=str,
                        help="Directory to save logs and models to")
    parser.add_argument("--continue_from_file",
                        help="Path to file containing previous training state.")
    parser.add_argument("--checkpoint_cycle", type=int, default=1,
                        help="Number of epochs between model checkpoints")
    parser.add_argument("--only_test", action="store_true",
                        help="Run only the test. --continue_from_file option "
                             "must also be set")
    args = parser.parse_args()

    with open(args.config_file, 'r') as infile:
        cfg = json.load(infile)

    if not os.path.isdir(args.overall_save_dir):
        os.makedirs(args.overall_save_dir)

    num_epoch = cfg["num_epoch"]

    for dataset in cfg["datasets"]:
        for base_type in cfg["models"]:
            for ec_k in cfg["k_vals"]:
                for loss_type in cfg["losses"]:
                    for enc, dec in cfg["enc_dec_types"]:
                        print(dataset, base_type,
                              ec_k, loss_type, enc, dec)
                        loss, loss_from_true_labels = get_loss(loss_type, cfg)
                        model_file, base, input_size, ds = get_base_model(
                            dataset, base_type)
                        parity_model, pm_input_size = get_parity_model(
                            dataset, base_type)

                        suffix_dir = os.path.join(dataset,
                                                  "{}".format(
                                                      base_type),
                                                  "k{}".format(ec_k),
                                                  "{}".format(loss_type),
                                                  "{}".format(enc),
                                                  "{}".format(dec))

                        save_dir = os.path.join(
                            args.overall_save_dir, suffix_dir)
                        config_map = get_config(num_epoch, ec_k, loss, enc,
                                                dec, model_file, base,
                                                ds, save_dir, input_size,
                                                parity_model, args.only_test,
                                                loss_from_true_labels,
                                                cfg)

                        if args.continue_from_file:
                            config_map["continue_from_file"] = args.continue_from_file

                        try:
                            trainer = ParityModelTrainer(config_map,
                                                         checkpoint_cycle=args.checkpoint_cycle)
                            trainer.train()
                        except KeyboardInterrupt:
                            print("INTERRUPTED")
