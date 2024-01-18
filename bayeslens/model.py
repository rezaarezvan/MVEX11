import torch.nn as nn


class B3Conv3FC(nn.Module):
    '''
    Simple Convolutional Neural Network
    '''

    def __init__(self, outputs, inputs, priors, layer_type, activation_type):
        super(B3Conv3FC, self).__init__()
        self.num_classes = outputs
        self.priors = priors
