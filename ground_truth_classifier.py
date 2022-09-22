from distutils.log import error
import numpy as np
from scipy.stats import norm
import torch, torch.nn


class NaiveBayes(torch.nn.Module):
    def __init__(self, num_features, num_classes, std):
        super(NaiveBayes, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.std = std

    def forward(self, x):

        try:
            mask = x[:,self.num_features:]
            x = x[:,:self.num_features]
        except IndexError:
            error("Classifier expects masking information to be concatenated with each feature vector.")

        y_classes = list(range(self.num_classes))

        output_probs = torch.zeros((len(x), self.num_classes))

        for y_val in y_classes:

            ## PDF values for each feature in x conditioned on the given label y_val

            # Default to PDF for U[0,1)
            p_x_y = torch.where((x >= 0) & (x < 1), torch.ones(x.shape), torch.zeros(x.shape))

            # Use normal distribution PDFs for appropriate features given y_val
            p_x_y[:,y_val:y_val+3] = torch.transpose(
                torch.Tensor(np.array([norm.pdf(x[:,y_val], y_val % 2, self.std), 
                        norm.pdf(x[:,y_val+1], (y_val // 2) % 2, self.std), 
                        norm.pdf(x[:,y_val+2], (y_val // 4) % 2, self.std)])), 0, 1)

            # Compute joint probability over masked features
            p_xo_y = torch.prod(torch.where(torch.gt(mask, 0), p_x_y, 1), dim=1)

            p_y = 1 / self.num_classes

            output_probs[:,y_val] = p_xo_y * p_y


        return torch.divide(output_probs, torch.squeeze(torch.dstack([torch.sum(output_probs, axis=1)]*self.num_classes)))


def true_classifier(x, mask=None, num_classes=8, std=0.3):

  if mask is None:
    mask = np.ones(x.shape)

  y_classes = list(range(num_classes))

  output_probs = np.zeros((len(x),num_classes))

  for y_val in y_classes:

    ## PDF values for each feature in x conditioned on the given label y_val

    # Default to PDF for U[0,1)
    p_x_y = np.where((x >= 0) & (x < 1), np.ones(x.shape), np.zeros(x.shape))

    # Use normal distribution PDFs for appropriate features given y_val
    p_x_y[:,y_val:y_val+3] = np.transpose(
        np.array([norm.pdf(x[:,y_val], y_val % 2, std), 
                  norm.pdf(x[:,y_val+1], (y_val // 2) % 2, std), 
                  norm.pdf(x[:,y_val+2], (y_val // 4) % 2, std)]))

    # Compute joint probability over masked features
    p_xo_y = np.array([np.prod(p_x_y[point][mask[point]==1]) for point in range(len(x))])

    p_y = 1 / num_classes

    output_probs[:,y_val] = p_xo_y * p_y


  return np.divide(output_probs, np.squeeze(np.dstack([np.sum(output_probs, axis=1)]*num_classes)))