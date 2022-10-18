import torch
import numpy as np

class PolicyEnvironment():
    def __init__(self, policy_network, classify_network, loss_function, alpha, stochastic=True):
        self.policy = policy_network
        self.classifier = classify_network
        self.loss_function = loss_function
        self.alpha = alpha
        self.stochastic = stochastic

    def rollout_batch(self, data, mask=None):

        (batch, labels) = data
        labels = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()
        batch_size = len(batch)
        num_features = len(batch[0])

        # Mask has shape (batch_size, num_actions) to make updates easier
        if mask is None:
            mask = torch.zeros((batch_size, num_features+1))

        output_data = []
        output_label = []
        output_action = []
        output_pred = []
        output_correct = []
        output_reward = []

        steps = 0

        while True and steps < 100:

            if len(batch.shape) == 1:
                batch = torch.unsqueeze(batch, 0)
                mask = torch.unsqueeze(mask, 0)
                labels = torch.unsqueeze(labels, 0)

            pred_vec =  torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            correct_vec = torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            reward_vec = torch.zeros(len(batch)).clone()

            action_probs = self.policy(torch.cat([batch, mask[:, :num_features]], -1))

            if self.stochastic:
                actions = torch.multinomial(action_probs, num_samples=1)
            else:
                actions = torch.argmax(action_probs, axis=1)

            action_mask = torch.nn.functional.one_hot(torch.squeeze(actions), num_classes=num_features+1)
            mask = torch.maximum(action_mask, mask)

            predict_bool = mask[:, -1] == 1
            acquire_bool = torch.logical_not(predict_bool)

            if torch.any(predict_bool):
                preds = self.classifier(torch.cat([batch[predict_bool, :], mask[predict_bool, :-1]], 1))
                loss = self.loss_function(preds, labels[predict_bool, :])

                pred_vec[predict_bool] = torch.argmax(preds, axis=1).float()
                correct_vec[predict_bool] = (torch.argmax(preds, axis=1) == torch.argmax(labels[predict_bool, :], axis=1)).float()
                reward_vec[predict_bool] = -loss
            reward_vec[acquire_bool] = -self.alpha

            output_data.append(batch)
            output_label.append(labels)
            output_action.append(torch.squeeze(actions, 1))
            output_pred.append(pred_vec)
            output_correct.append(correct_vec)
            output_reward.append(reward_vec)

            if not torch.any(acquire_bool):
                break

            batch = torch.squeeze(batch[acquire_bool, :])
            mask = torch.squeeze(mask[acquire_bool, :])
            labels = torch.squeeze(labels[acquire_bool, :])
            steps += 1

        return [torch.cat(output, 0) for output in [output_data, output_label, output_action, output_pred, output_correct, output_reward]]


        