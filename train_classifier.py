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


if __name__ == "__main__":

    data_path = "./cube_20_0.3.pkl"
    batch_size = 32
    epochs = 20
    expected_features = 10
    total_features = 20

    train_dset = CubeDataset(data_path, split="train")
    val_dset = CubeDataset(data_path, split="valid")
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size)

    cube_classifier = Net(num_features=total_features*2, num_outputs=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cube_classifier.parameters(), lr=0.01)

    for e in range(epochs):
        for batch, (data, labels) in enumerate(train_loader):

            if batch % 100 == 0:
                print("Batch", batch)

            y = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()

            optimizer.zero_grad()

            mask = torch.floor(torch.rand(data.shape) + expected_features / total_features)

            pred_y = cube_classifier(torch.cat((torch.mul(data, mask), mask), 1))
            loss = criterion(pred_y, y)
            loss.backward()

            optimizer.step()

        val_mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

        val_preds = torch.argmax(cube_classifier(torch.cat((torch.mul(torch.Tensor(val_dset.x), val_mask), val_mask), 1)), axis=-1)
        num_correct = torch.sum(torch.eq(val_preds, torch.Tensor(val_dset.y)))
        print("Validation Accuracy:", num_correct / len(val_preds))
    
    mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

    bayes = NaiveBayes(total_features, 8, 0.3)

    bayes_preds = torch.argmax(bayes(torch.cat((torch.mul(torch.Tensor(val_dset.x), mask), mask), 1)), axis=-1)

    print(bayes_preds[0:20])

    num_correct = torch.sum(torch.eq(bayes_preds, torch.Tensor(val_dset.y)))

    print("Naive Bayes Accuracy:", num_correct/len(bayes_preds))