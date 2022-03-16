import sys
import time
import json
import logging
from datetime import datetime

import torch
from torch import nn

from data_loader import SlidingWindowDataset, create_data_loaders, load_from_file
from models import Graph_Transformer
from trainer import Trainer
from utils_config import create_dirs, get_args, process_config

torch.manual_seed(0)

def main():
    start = time.perf_counter()

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        logging.error(Ex)
        logging.error("Missing or invalid config file.")
        sys.exit(1)

    config['timestamp_training'] = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")

    logging.info(args)
    logging.info(json.dumps(config, indent=4, separators=(',', ': ')))

    create_dirs(config["result_dir"], config["model_dir"])

    x_train = load_from_file(config["data_path"])
    train_dataset = SlidingWindowDataset(x_train, config["window_size"])
    train_loader, val_loader, _ = create_data_loaders(train_dataset=train_dataset,
                                                      test_dataset=None,
                                                      batch_size=config["batch_size"],
                                                      val_split=config["val_split"],
                                                      shuffle=bool(config["shuffle"]),
                                                      num_workers=config["dataloader_num_workers"])

    logging.info(f'Loaded training dataset in {time.perf_counter() - start} seconds.')

    model = Graph_Transformer(n_features=config["n_features"],
                              window_size=config["window_size"],
                              kernel_size=config["kernel_size"],
                              feat_gat_embed_dim=config["feature_embed_dim"],
                              use_gatv2=config["use_gatv2"],
                              dropout=config["dropout"],
                              alpha=config["alpha"],
                              N=config["num_transformer_stacks"],
                              d_ff=config["d_ff"],
                              h=config["num_heads"])

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    trainer = Trainer(model_name=config["model"],
                      model=model,
                      optimizer=opt,
                      criterion=criterion,
                      config=config)
    trainer.train(train_loader, val_loader)
    trainer.save_results()
    logging.info(f'Finish training script in {time.perf_counter() - start} seconds.')

if __name__ == "__main__":
    main()
