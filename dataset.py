import pickle
import torch

class CubeDataset(torch.utils.data.Dataset):
    def __init__(self, fname, split="train"):
        with open(fname, "rb") as f:
            data_obj = pickle.load(f)
            data_tuple = data_obj[split]
            self.x = data_tuple[0]
            self.y = data_tuple[1]
            self.data = list(zip(self.x, self.y))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.data[index]

class BehaviorDataset(torch.utils.data.Dataset):
    def __init__(self, fname, split, split_point):
        with open(fname, "rb") as f:
            data_obj = pickle.load(f)
            full_len = len(data_obj["X"])

            if(split_point == "train"):
                start_idx = 0
                end_idx = int(split * full_len)
            else:
                start_idx = int(split * full_len)
                end_idx = -1

            self.x = data_obj["X"][start_idx:end_idx]
            self.mask = data_obj["mask"][start_idx:end_idx]
            self.y = data_obj["Action"][start_idx:end_idx]
            self.data = list(zip(self.x, self.mask, self.y))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.data[index]
