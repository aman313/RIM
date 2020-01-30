import torch
from allennlp.common import Registrable
from allennlp.models import Model


class RIMModule(torch.nn.Module,Registrable):

    def __init__(self, hidden_state_size, num_kernels, active_kernels, dynamics_module, attention_module):
        pass

    def reset_hidden(self, init_hidden):
        self._current_hidden = init_hidden

    def get_update_mask(self, topk_indices):
        pass

    def get_attended_input(self, value, similarity_matrix, update_mask):
        pass

    def forward(self, input, query_suffix=None):
        '''

        :param input_sequence:
        :param prev_state:
        :return:
            use hidden state of each rim as query and get attention weights and attended input representation
            select top k kernels by attention weight
            run one step of dynamics for selected kernels
            update by communincation with other rims
        '''

        #concatenate null row to the sequence: batch_size X seq_len X dimension -> batch_size X seq_len+1 X dimension
        null_tensor = torch.zeros((input.shape(0),1,input.shape(2)), device=input.device)
        null_concated_input = torch.cat([input,null_tensor],dim=1)
        queries = self._rim_hidden_states * self._hidden_to_query_map # num_kernels X hidden_state_size * hidden_state_size X attention_dim -> num_kernels X attention_dim
        values = self._rim_hidden_states * self._hidden_to_values_map # num_kernels X hidden_state_size * hidden_state_size X attention_dim -> num_kernels X attention_dim
        keys = self._input_to_key_map(input) # batch_size X seq_len+1 X dimension * dimension X attention_dim -> batch_size X seq_len+1 X attention_dim
        expanded_queries = queries.expand(null_concated_input.shape(0),-1,-1) # num_kernels X attention_dim -> batch_size X num_kernels X attention_dim
        attention_similarity_matrix = self._attention_module(keys,expanded_queries) # -> batch_size X num_kernsl X seq_len +1
        topk_vals, topk_indices = torch.topk(attention_similarity_matrix[:,:,:,-1], self._active_kernels,largest=False) # batch_size X num_kernels X 1

        #maybe remove the null dimension from the input?
        update_mask = self.get_update_mask(topk_indices)
        attended_input = self.get_attended_input(values,attention_similarity_matrix,update_mask)
        updated_hidden = self._dynamics_module(self._current_hidden, attended_input,update_mask)
        self._current_hidden = updated_hidden
        return self._current_hidden


class RIMEncoder(Model):
    pass


class RIMDecoder(Model):
    pass
