import os
import time
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from utils_config import save_config

class Trainer:
    def __init__(self, model_name, model, optimizer, criterion, config):
        self.model_name = model_name
        self.model = model
        self.optimizer = deepcopy(optimizer)
        self.criterion = deepcopy(criterion)
        self.mask = self._create_mask(config)
        self.n_epochs = config["num_epochs"]
        self.device = torch.device("cuda:0" if config["device"] == "gpu" and torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_initialized():
            torch.cuda.init()
        self.model.to(self.device)
        self.model.float()
        self.mask.to(self.device)
        self.losses = { "train": [], "val": [] }
        self.training_time_per_epoch = []
        self.total_training_time = None
        self.model_dir = config['model_dir']
        self.best_epoch = 0 # starting index is 1
        self.config = config

    def train(self, train_loader, val_loader=None):
        start = time.perf_counter()
        logging.info(f"----- START TRAINING THE {self.model_name.upper()} FOR {self.n_epochs} EPOCHS -----")
        for epoch in range(1, self.n_epochs + 1):
            logging.info(f"Training epoch {epoch}...")
            self.train_epoch(epoch, train_loader, val_loader)
        self.total_training_time = time.perf_counter() - start
        logging.info(f"----- COMPLETED TRAINING THE {self.model_name.upper()} IN {self.total_training_time} seconds -----")

    def train_epoch(self, epoch, train_loader, val_loader=None):
        start = time.perf_counter()
        b_losses = []
        self.model.train()

        for batch in train_loader:
            src = batch.float()
            src = src.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(src, src_mask=self.mask)

            assert out.size(1) == src.size(1)
            train_loss = torch.sqrt(self.criterion(src, out))
            train_loss.backward()
            self.optimizer.step()
            b_losses.append(train_loss.item())

        b_losses = np.array(b_losses)
        train_loss = np.sqrt((b_losses ** 2).mean())
        self.losses["train"].append(train_loss)

        base_output = f'Model {self.model_name.upper()}. Local Training Epoch: {epoch} \t'
        logging.info(base_output + f'Train Loss: {train_loss:.6f}')

        # Evaluate on validation set
        val_loss = "NA"
        if val_loader is not None:
            val_loss = self.evaluate(val_loader)
            self.losses["val"].append(val_loss)

            if val_loss <= self.losses["val"][-1]:
                self.save(f"{self.model_name}.pt")
                self.best_epoch = epoch
            logging.info(base_output + f'Validation Loss: {val_loss:.6f}')
        else:
            if train_loss <= self.losses["train"][-1]:
                self.save(f"{self.model_name}.pt")
                self.best_epoch = epoch

        epoch_time = time.perf_counter() - start
        self.training_time_per_epoch.append(epoch_time)
        logging.info(base_output + f'Epoch Training Time: {epoch_time} seconds')
        logging.info('-' * (len(base_output) + 30))

    def evaluate(self, data_loader):
        b_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                src = batch.float()
                src = src.to(self.device)
                out = self.model(src, src_mask=self.mask)
                loss = torch.sqrt(self.criterion(src, out))
                b_losses.append(loss.item())

        b_losses = np.array(b_losses)
        loss = np.sqrt((b_losses ** 2).mean())
        return loss

    def infer_loss(self, test_loader, n_test):
        """
        Only used in inference phase. Compute recons loss in a slightly different manner.
        """
        self.model.eval()
        recon_loss = np.zeros(n_test)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_size = batch.size(0)
                src = batch.float()
                src = src.to(self.device)
                out = self.model(src, src_mask=self.mask)
                for j in range(batch_size):
                    recon_loss[i * batch_size + j] = torch.sqrt(self.criterion(out[j, self.config["pre_mask"]:self.config["post_mask"], :],
                                                                               src[j, self.config["pre_mask"]:self.config["post_mask"], :]))
        return recon_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.model_dir + file_name
        if os.path.exists(self.model_dir):
            pass
        else:
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def save_results(self) -> dict:
        # Save config
        self.config['best_epoch'] = self.best_epoch
        self.config['total_training_time'] = self.total_training_time
        save_config(self.config)

        # Save loss
        df = pd.DataFrame(self.losses)
        df['training_time'] = self.training_time_per_epoch
        df.insert(0, "epoch", [i for i in range(1, self.n_epochs + 1)], True)
        df.to_csv(f"{self.config['result_dir']}epoch_loss_{self.model_name}.csv",
                  index=False, header=['Epoch', 'TrainingLoss', 'ValidationLoss', 'TrainingTime'])

        return self.config

    def plot_loss(self):
        pass

    def _create_mask(self, config):
        mask = torch.ones(1, config["window_size"], config["window_size"])
        mask[:, config["pre_mask"]:config["post_mask"], :] = 0
        mask[:, :, config["pre_mask"]:config["post_mask"]] = 0
        return mask
