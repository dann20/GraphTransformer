import logging

import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class SlidingWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int, horizon=1):
        if len(data.shape) == 1:
            data = np.expand_dims(data, -1)
        self.data = data
        self.window = window
        self.horizon = horizon
        logging.info(f'Created SlidingWindowDataset with shape {self.shape}')

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window + 1

def load_from_file(path: str):
    data = np.load(path)
    return data

def create_data_loaders(train_dataset=None, test_dataset=None, batch_size=128, val_split=0.1, shuffle=True, num_workers=1):
    train_loader, val_loader, test_loader = None, None, None
    if train_dataset is not None:
        if val_split == 0.0:
            logging.info(f"train_size: {len(train_dataset)}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        else:
            dataset_size = len(train_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(val_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
            val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

            logging.info(f"train_size: {len(train_indices)}")
            logging.info(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logging.info(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
