from dense_classifier import Net
from ground_truth_classifier import NaiveBayes
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

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


if __name__ == "__main__":

    # data_path = "./cube_20_0.3.pkl"
    data_path = "../afa-imitation-learning/data/behav_clone.pkl"
    batch_size = 32
    epochs = 20
    expected_features = 10
    total_features = 20

    mode = "behavior"

    sp = 0.8

    if mode == "cube":
        train_dset = CubeDataset(data_path, split="train")
        val_dset = CubeDataset(data_path, split="valid")
        neural_classifier = Net(num_features=total_features*2, num_outputs=8)
    else:
        train_dset = BehaviorDataset(data_path, split=sp, split_point="train")
        val_dset = BehaviorDataset(data_path, split=sp, split_point="val")
        neural_classifier = Net(num_features=total_features*2, num_outputs=21)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(neural_classifier.parameters(), lr=0.1)

    for e in range(epochs):
        for batch_idx, batch in enumerate(train_loader):

            if mode == "cube":
                (data, labels) = batch
                y = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()
                mask = torch.floor(torch.rand(data.shape) + expected_features / total_features)
                data = torch.mul(data, mask)
            else:
                (data, mask, labels) = batch
                y = torch.nn.functional.one_hot(labels.long(), num_classes=21).float()

            if batch_idx % 100 == 0:
                print("Batch", batch_idx)          

            optimizer.zero_grad()

            pred_y = neural_classifier(torch.cat((data, mask), 1).float())
            loss = criterion(pred_y, y)
            loss.backward()

            optimizer.step()

        val_mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

        val_preds = torch.argmax(neural_classifier(torch.cat((torch.mul(torch.Tensor(val_dset.x), val_mask), val_mask), 1)), axis=-1)
        num_correct = torch.sum(torch.eq(val_preds, torch.Tensor(val_dset.y)))
        print("Validation Accuracy:", num_correct / len(val_preds))
    
    if mode == "cube":
        mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

        bayes = NaiveBayes(total_features, 8, 0.3)

        bayes_preds = torch.argmax(bayes(torch.cat((torch.mul(torch.Tensor(val_dset.x), mask), mask), 1)), axis=-1)

        print(bayes_preds[0:20])

        num_correct = torch.sum(torch.eq(bayes_preds, torch.Tensor(val_dset.y)))

        print("Naive Bayes Accuracy:", num_correct/len(bayes_preds))