import torch
from allennlp.models import Model


class RIMModule(torch.nn.Module):

    def __init__(self, num_kernels, active_kernels, dynamics_module):
        raise NotImplementedError

    def forward(self, input_sequence, prev_state, query_suffix=None):
        '''

        :param input_sequence:
        :param prev_state:
        :return:
            use hidden state of each rim as query and get attention weights and attended input representation
            select top k kernels by attention weight
            run one step of dynamics for selected kernels
            update by communincation with other rims
        '''
        raise NotImplementedError


class RIMEncoder(Model):
    pass


class RIMDecoder(Model):
    pass
