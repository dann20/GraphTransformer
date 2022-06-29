import sys
import time
import json
import logging
from datetime import datetime

import numpy as np
import torch
from torch import nn

from data_loader import SlidingWindowDataset, create_data_loaders, load_from_file
from models import Graph_Transformer
from trainer import Trainer
from utils_config import get_args, process_config, get_config_from_json, save_config
from utils_inference import create_labels, select_KQp_threshold, select_threshold

torch.manual_seed(10)
np.random.seed(0)


def heuristic_inference(recon_loss, anomaly_index, test_labels, config, n_threshold=20):
    logging.info("Inference with heuristic method.")
    config["threshold_method"] = "heuristic"
    best_thr, auc, accuracy, precision, recall, F1 = select_threshold(
        recon_loss, anomaly_index, test_labels, config, n_threshold
    )
    config["best_threshold"] = best_thr
    config["accuracy"] = accuracy
    config["precision"] = precision
    config["recall"] = recall
    config["F1"] = F1
    config["AUC"] = auc
    return config


def KQE_inference(recon_loss, anomaly_index, test_labels, config):
    logging.info("Inference with KQE.")
    config["threshold_method"] = "KQE"
    q_best, accuracy, precision, recall, F1 = select_KQp_threshold(
        recon_loss, anomaly_index, test_labels
    )
    config["q_best"] = q_best
    config["accuracy"] = accuracy
    config["precision"] = precision
    config["recall"] = recall
    config["F1"] = F1
    return config


@torch.no_grad()
def main():
    start = time.perf_counter()
    fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as ex:
        logging.error(ex)
        logging.error("Missing or invalid arguments")
        sys.exit(1)

    timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")

    filename = config["result_dir"] + "training_config.json"

    try:
        config = get_config_from_json(filename)
        config["timestamp_inference"] = timestamp
        logging.info(args)
        logging.info(json.dumps(config, indent=4, separators=(",", ": ")))
    except Exception as ex:
        logging.error(ex)
        logging.error("No config found in result_dir.")
        sys.exit(1)

    x_test = load_from_file(config["data_path"], file="test")
    test_dataset = SlidingWindowDataset(x_test, config["window_size"])
    _, _, test_loader = create_data_loaders(
        train_dataset=None,
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        shuffle=bool(config["shuffle"]),
        num_workers=config["dataloader_num_workers"],
    )

    logging.info(f"Loaded training dataset in {time.perf_counter() - start} seconds.")

    trained_model = Graph_Transformer(
        n_features=config["n_features"],
        window_size=config["window_size"],
        kernel_size=config["kernel_size"],
        feat_gat_embed_dim=config["feature_embed_dim"],
        use_gatv2=config["use_gatv2"],
        dropout=config["dropout"],
        alpha=config["alpha"],
        N=config["num_transformer_stacks"],
        d_ff=config["d_ff"],
        h=config["num_heads"],
    )

    criterion = nn.MSELoss()
    trainer = Trainer(
        model_name=config["model"],
        model=trained_model,
        optimizer=None,
        criterion=criterion,
        config=config,
    )
    trainer.load(config["model_dir"] + f"{trainer.model_name}.pt")
    logging.info(f"Loaded model {trainer.model_name}.")

    start_inference = time.perf_counter()
    logging.info("----------- STARTING INFERENCE PHASE... ------------")
    n_test = len(test_dataset)
    logging.info(f"n_test = {n_test}")
    recon_loss = trainer.infer_loss(test_loader, n_test)
    idx_anomaly_test = load_from_file(config["data_path"], file="idx_anomaly_test")
    anomaly_index, test_labels = create_labels(idx_anomaly_test, n_test, config)

    config = heuristic_inference(
        recon_loss, anomaly_index, test_labels, config, n_threshold=20
    )

    config["total_inference_time"] = time.perf_counter() - start_inference
    save_config(config)
    logging.info(f"Total inference time: {config['total_inference_time']} seconds.")
    logging.info("----------- COMPLETED INFERENCE PHASE. ------------")
    # logging.info("TP: {}".format(n_TP))
    # logging.info("FP: {}".format(n_FP))
    # logging.info("FN: {}".format(n_FN))


if __name__ == "__main__":
    main()
