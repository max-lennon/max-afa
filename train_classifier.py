from dense_classifier import Net
from ground_truth_classifier import NaiveBayes
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import BehaviorDataset, CubeDataset
from policy_rollout_env import PolicyEnvironment

def train_network(train_data, val_data, batch_size, epochs, mode, criterion=None, optimizer_func=None, expected_features=None):
    total_features = 20

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer_func is None:
        optimizer_func = torch.optim.SGD

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    if mode == "cube":
        neural_classifier = Net(num_features=total_features*2, num_outputs=8)
    else:
        neural_classifier = Net(num_features=total_features*2, num_outputs=21)

    optimizer = optimizer_func(neural_classifier.parameters(), lr=0.1)

    for e in range(epochs):

        neural_classifier.softmax = False

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

        total_correct = 0

        neural_classifier.softmax = True

        for batch_idx, batch in enumerate(val_loader):

            if mode == "cube":
                (data, labels) = batch
                y = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()
                val_mask = torch.floor(torch.rand(data.shape) + expected_features / total_features)
                data = torch.mul(data, mask)
            else:
                (data, val_mask, labels) = batch
                y = torch.nn.functional.one_hot(labels.long(), num_classes=21).float()

            # val_mask = torch.floor(torch.rand(data.shape) + expected_features / total_features)

            val_preds = torch.argmax(neural_classifier(torch.cat((torch.mul(data, val_mask), val_mask), 1).float()), axis=-1)
            num_correct = torch.sum(torch.eq(val_preds, labels))
            total_correct += num_correct

        print("Validation Accuracy:", total_correct / len(val_data))

    return neural_classifier

if __name__ == "__main__":

    # data_path = "./cube_20_0.3.pkl"
    behav_data_path = "../afa-imitation-learning/data/behav_clone.pkl"
    cube_data_path = "./cube_20_0.3.pkl"
    batch_size = 32
    epochs = 2
    expected_features = 10
    total_features = 20

    mode = "behavior"

    sp = 0.8

    if mode == "cube":
        train_dset = CubeDataset(cube_data_path, split="train")
        val_dset = CubeDataset(cube_data_path, split="valid")
    else:
        train_dset = BehaviorDataset(behav_data_path, split=sp, split_point="train")
        val_dset = BehaviorDataset(behav_data_path, split=sp, split_point="val")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD

    classifier = train_network(train_dset, val_dset, batch_size, epochs, mode, criterion, optimizer)

    if mode == "cube":
        mask = torch.floor(torch.rand(val_dset.x.shape) + expected_features / total_features)

        bayes = NaiveBayes(total_features, 8, 0.3)

        bayes_preds = torch.argmax(bayes(torch.cat((torch.mul(torch.Tensor(val_dset.x), mask), mask), 1)), axis=-1)

        print(bayes_preds[0:20])

        num_correct = torch.sum(torch.eq(bayes_preds, torch.Tensor(val_dset.y)))

        print("Naive Bayes Accuracy:", num_correct/len(bayes_preds))

    cube_loader = torch.utils.data.DataLoader(CubeDataset(cube_data_path, split="train"), batch_size=3)

    env = PolicyEnvironment(classifier, NaiveBayes(total_features, 8, 0.3), criterion, 0.05)
    for batch_idx, batch in enumerate(cube_loader):
        if batch_idx > 0:
            break
        print(env.rollout_batch(batch))


    