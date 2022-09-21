from dense_classifier import Net
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


if __name__ == "__main__":

    data_path = "./cube_20_0.3.pkl"
    batch_size = 16
    epochs = 100
    expected_features = 15
    total_features = 20

    train_dset = CubeDataset(data_path, split="train")
    val_dset = CubeDataset(data_path, split="valid")
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

    cube_classifier = Net(num_features=total_features, num_outputs=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cube_classifier.parameters(), lr=0.01)

    for e in range(epochs):
        for batch, (data, labels) in enumerate(train_loader):

            if batch % 100 == 0:
                print("Batch", batch)

            y = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()

            optimizer.zero_grad()

            mask = torch.floor(torch.rand(data.shape) + expected_features / total_features)

            pred_y = cube_classifier(torch.mul(data, mask))
            loss = criterion(pred_y, y)
            loss.backward()

            optimizer.step()

        val_mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

        val_preds = torch.argmax(cube_classifier(torch.mul(torch.Tensor(val_dset.x), val_mask)), axis=-1)
        num_correct = torch.sum(torch.eq(val_preds, torch.Tensor(val_dset.y)))
        print("Validation Accuracy:", num_correct / len(val_preds))