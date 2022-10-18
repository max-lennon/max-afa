import torch
import numpy as np

"""
    A class to enact an active feature acquisition policy consisting of a behavior policy network and a classifier.

    ...

    Attributes
    ----------
    policy_network : func
        A callable function that takes in masked data + mask (concatenated) information and outputs a sequence of probabilities corresponding to each possible action (i.e. acquiring 1-n features or making a prediction)
    
    classify_network : func
        A callable function that takes in masked data + mask (concatenated) and outputs a prediction of the class of the data example.
                    
    loss_function : func
        A callable loss function, e.g. CrossEntropyLoss.
    
    alpha : float 
        The reward penalty for acquiring a new feature.
                    
    stochastic : bool
        Whether to sample the action probabilities in a weighted fashion or take the action with highest probability every time.

    Methods
    -------
    rollout_batch(data, mask, hide_val):
        Returns a dataset consisting of data, label, action, prediction, correct, and reward tensors.
"""
class PolicyEnvironment():
    def __init__(self, policy_network, classify_network, loss_function, alpha, stochastic=True):
        self.policy = policy_network
        self.classifier = classify_network
        self.loss_function = loss_function
        self.alpha = alpha
        self.stochastic = stochastic

    def rollout_batch(self, data, mask=None, hide_val=10):

        (batch, labels) = data
        labels = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()
        batch_size = len(batch)
        num_features = len(batch[0])

        # Mask has shape (batch_size, num_actions) to make updates easier
        if mask is None:
            mask = torch.zeros((batch_size, num_features+1))


        # Temporary structures to hold each time-sliced component of the final dataset; each list will be concatenated across time steps at the end.
        output_data = []
        output_label = []
        output_action = []
        output_pred = []
        output_correct = []
        output_reward = []

        steps = 0

        # Stopping condition to prevent possible infinite loops; we expect that this loop should terminate much more quickly
        while steps < 100:

            # Prevents shape conflicts in the case of one example left in the batch to analyze
            if len(batch.shape) == 1:
                batch = torch.unsqueeze(batch, 0)
                mask = torch.unsqueeze(mask, 0)
                labels = torch.unsqueeze(labels, 0)


            pred_vec =  torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            correct_vec = torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            reward_vec = torch.zeros(len(batch)).clone()

            # Mask data and pass to policy network to obtain action probabilities (NOTE: output of policy network is never allowed to be < 0)
            feature_mask = mask[:, :num_features]
            action_probs = self.policy(torch.cat([torch.mul(batch, feature_mask) - (1-feature_mask) * hide_val, feature_mask], -1))

            if self.stochastic:
                actions = torch.multinomial(action_probs, num_samples=1)
            else:
                actions = torch.argmax(action_probs, axis=1)

            # Update ongoing feature acquisition mask according to the action selected above (NOTE: full mask has an entry for taking the prediction action)
            action_mask = torch.nn.functional.one_hot(torch.squeeze(actions), num_classes=num_features+1)
            mask = torch.maximum(action_mask, mask)

            # For smart indexing: get a vector showing which examples in the batch have been selected to predict on (and opposite, for convenience)
            predict_bool = mask[:, -1] == 1
            acquire_bool = torch.logical_not(predict_bool)

            if torch.any(predict_bool):
                preds = self.classifier(torch.cat([torch.mul(batch[predict_bool, :], mask[predict_bool, :-1]), mask[predict_bool, :-1]], 1))
                loss = self.loss_function(preds, labels[predict_bool, :])

                # Results in a vector of same length as batch where predictions show up as the predicted value, and no predictions show up as nan
                pred_vec[predict_bool] = torch.argmax(preds, axis=1).float()

                # Results in a vector of same length as batch where wrong predictions show up as 0, correct predictions as 1, and no predictions as nan
                correct_vec[predict_bool] = (torch.argmax(preds, axis=1) == torch.argmax(labels[predict_bool, :], axis=1)).float()

                reward_vec[predict_bool] = -loss
            
            if torch.any(acquire_bool):
                reward_vec[acquire_bool] = -self.alpha

            output_data.append(batch)
            output_label.append(labels)
            output_action.append(torch.squeeze(actions, 1))
            output_pred.append(pred_vec)
            output_correct.append(correct_vec)
            output_reward.append(reward_vec)

            # Terminate loop if no data points are left to predict
            if not torch.any(acquire_bool):
                break

            # Repeat the policy on only the examples that have not been predicted yet 
            # (NOTE: this means that batch changes shape, and that batch_size is not necessarily the same as len(batch) at all times)
            batch = torch.squeeze(batch[acquire_bool, :])
            mask = torch.squeeze(mask[acquire_bool, :])
            labels = torch.squeeze(labels[acquire_bool, :])
            steps += 1

        return [torch.cat(output, 0) for output in [output_data, output_label, output_action, output_pred, output_correct, output_reward]]


        