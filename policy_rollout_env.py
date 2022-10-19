import torch

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
    rollout_batch(data, hide_val):
        Returns a behavior cloning dataset consisting of data, label, action, prediction, correct, and reward tensors for a given batch of examples.
    
    rollout_dataset(data, batch_size, hide_val):
        Takes in a classification dataset and divides it into batches, then calls rollout_batch on each batch and concatenates the outputs into a final behavior dataset.

"""
class PolicyEnvironment():
    def __init__(self, policy_network, classify_network, loss_function, alpha, fixed_batch=False, logits=True, repeat_action=False, stochastic=True):
        self.policy = policy_network
        self.classifier = classify_network
        self.loss_function = loss_function
        self.alpha = alpha
        self.fixed_batch = fixed_batch
        self.logits = logits
        self.repeat_action = repeat_action
        self.stochastic = stochastic
        

    def rollout_batch(self, data, hide_val=-10):

        (batch, labels) = data
        labels = torch.nn.functional.one_hot(labels.long(), num_classes=8).float()
        batch_size = len(batch)
        num_features = len(batch[0])

        # Mask has shape (batch_size, num_actions) to make updates easier
        mask = torch.zeros((batch_size, num_features+1))
        feature_counts = torch.zeros((batch_size, num_features))


        # Temporary structures to hold each time-sliced component of the final dataset; each list will be concatenated across time steps at the end.
        output_data = []
        output_label = []
        output_action = []
        output_counts = []
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
                feature_counts = torch.unsqueeze(feature_counts, 0)
                labels = torch.unsqueeze(labels, 0)


            pred_vec =  torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            correct_vec = torch.broadcast_to(torch.Tensor([float("nan")]), (len(batch),)).clone()
            reward_vec = torch.zeros(len(batch)).clone()

            # Mask data and pass to policy network to obtain action probabilities (NOTE: output of policy network is never allowed to be < 0)
            feature_mask = mask[:, :num_features]

            if self.logits:
                action_logits = self.policy(torch.cat([torch.mul(batch, feature_mask) + (1-feature_mask) * hide_val, feature_mask], -1))
                action_probs = torch.nn.Softmax()(action_logits)
            else:
                action_probs = self.policy(torch.cat([torch.mul(batch, feature_mask) + (1-feature_mask) * hide_val, feature_mask], -1))

            # Zero out probabilities for actions that have already been taken if applicable
            if not self.repeat_action:
                action_probs = torch.mul(1-mask, action_probs)
                action_probs = torch.maximum(action_probs, torch.Tensor([1e-6]))
                action_probs = torch.div(action_probs, torch.unsqueeze(torch.sum(action_probs, axis=1), 1))

            if self.stochastic:
                actions = torch.multinomial(action_probs, num_samples=1)
            else:
                actions = torch.argmax(action_probs, axis=1)

            # Update ongoing feature acquisition mask according to the action selected above (NOTE: full mask has an entry for taking the prediction action)
            action_mask = torch.nn.functional.one_hot(torch.squeeze(actions), num_classes=num_features+1)
            mask = torch.maximum(action_mask, mask)

            if len(action_mask.shape) == 1:
                action_mask = torch.unsqueeze(action_mask, 0)

            feature_counts += action_mask[:, :num_features]

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
            output_counts.append(feature_counts)
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
            feature_counts = torch.squeeze(feature_counts[acquire_bool, :])
            labels = torch.squeeze(labels[acquire_bool, :])
            steps += 1

        return [torch.cat(output, 0) for output in [output_data, output_label, output_action, output_counts, output_pred, output_correct, output_reward]]


    def rollout_dataset(self, data, batch_size, hide_val=-10):
        num_batches = len(data) // batch_size

        full_dataset = self.rollout_batch(data[0:batch_size])

        for i in range(1, num_batches):
            batch_dataset = self.rollout_batch(data[batch_size*i:batch_size*(i+1)])
            full_dataset = [torch.cat([full_dataset[e], batch_dataset[e]], 0) for e in len(full_dataset)]

        if num_batches * batch_size != len(data) and not self.fixed_batch:
            batch_dataset = self.rollout_batch(data[batch_size*num_batches:-1])
            full_dataset = [torch.cat([full_dataset[e], batch_dataset[e]], 0) for e in len(full_dataset)]

        return full_dataset
        
