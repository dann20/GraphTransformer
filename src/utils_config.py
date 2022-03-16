import argparse
import json
import os

model_default_config = {
    "kernel_size": 7,
    "feature_embed_dim": None,
    "use_gatv2": True,
    "alpha": 0.2,
    "window_size": 50,
    "batch_size": 128,
    "lr": 0.001,
    "num_transformer_stacks": 2,
    "pre_mask": 20,
    "post_mask": 30,
    "d_ff": 128,
    "num_heads": 1,
    "dropout": 0.1,
    "num_epochs": 10,
    "shuffle": 1,
    "val_split": 0.0,
    "dataloader_num_workers": 1,
    "device": "cpu",
}

def get_config_from_json(json_file) -> dict:
    """
    Get the config from a json file
    :param: json_file
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    return config_dict

def save_config(config):
    filename = config["result_dir"] + "training_config.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def process_config(json_file):
    config = get_config_from_json(json_file)
    for key, value in model_default_config.items():
        _ = config.setdefault(key, value)

    # create directories to save experiment results and trained models
    save_dir = "../experiments/{}/{}".format(config["experiment"], config["dataset"])
    config["result_dir"] = os.path.join(save_dir, "results/")
    config["model_dir"] = os.path.join(save_dir, "models/")
    config["data_path"] = f"../data/{config['data_dir']}/{config['dataset']}.npz"
    return config

def create_dirs(*dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args
