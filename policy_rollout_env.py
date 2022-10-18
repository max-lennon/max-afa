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

        while True and steps < 21:

            pred_vec =  torch.broadcast_to(torch.Tensor(float("nan")), (batch_size))
            correct_vec = torch.broadcast_to(torch.Tensor(float("nan")), (batch_size))
            reward_vec = torch.zeros(batch_size)

            action_probs = self.policy(torch.cat([batch, mask[:, :num_features]], 1))

            if self.stochastic:
                actions = torch.multinomial(action_probs, num_samples=1)
            else:
                actions = torch.argmax(action_probs, axis=1)

            mask[:, actions] = 1

            predict_bool = torch.prod(mask[:, -1]) == 1

            predict_idx = predict_bool.nonzero()
            acquire_idx = torch.logical_not(predict_bool).nonzero()

            preds = self.classifier(batch[predict_idx])
            loss = self.loss_function(preds, labels[predict_idx])

            pred_vec[predict_idx] = preds
            correct_vec[predict_idx] = preds == labels[predict_idx]

            reward_vec[predict_idx] = -loss
            reward_vec[acquire_idx] = -self.alpha

            output_data.append(batch)
            output_label.append(labels)
            output_action.append(actions)
            output_pred.append(pred_vec)
            output_correct.append(correct_vec)
            output_reward.append(reward_vec)

            if len(acquire_idx) == 0:
                break

            batch = batch[acquire_idx]
            label = label[acquire_idx]
            steps += 1

        return [torch.cat(output) for output in [output_data, output_label, output_action, output_pred, output_correct, output_reward]]


        