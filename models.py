import torch
from allennlp.common import Registrable
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules.seq2seq_decoders import SeqDecoder


class RecurrentDynamicsModule(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()

@RecurrentDynamicsModule.register('lstm_dynamics')
class LSTMDynamicsModule(RecurrentDynamicsModule):

    def __init__(self, input_dim, hidden_dim, num_kernels):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_kernels = num_kernels
        self._input_to_forget = torch.nn.Parameter(torch.rand(num_kernels ,input_dim, hidden_dim), requires_grad=True)
        self._hidden_to_forget = torch.nn.Parameter(torch.rand(num_kernels, hidden_dim, hidden_dim), requires_grad=True)
        self._forget_bias = torch.nn.Parameter(torch.zeros(num_kernels, hidden_dim), requires_grad=True)
        self._input_to_input = torch.nn.Parameter(torch.rand(num_kernels, input_dim,hidden_dim), requires_grad=True)
        self._hidden_to_input = torch.nn.Parameter(torch.rand( num_kernels, hidden_dim,hidden_dim), requires_grad=True)
        self._input_bias = torch.nn.Parameter(torch.zeros(num_kernels, hidden_dim), requires_grad=True)
        self._input_to_out = torch.nn.Parameter(torch.rand(num_kernels, input_dim,hidden_dim), requires_grad=True)
        self._hidden_to_out = torch.nn.Parameter(torch.rand(num_kernels, hidden_dim, hidden_dim), requires_grad=True)
        self._out_bias = torch.nn.Parameter(torch.zeros(num_kernels, hidden_dim), requires_grad=True)
        self._input_to_cell = torch.nn.Parameter(torch.rand(num_kernels, input_dim,hidden_dim), requires_grad=True)
        self._hidden_to_cell = torch.nn.Parameter(torch.rand(num_kernels, hidden_dim, hidden_dim), requires_grad = True)
        self._cell_bias = torch.nn.Parameter(torch.zeros(num_kernels, hidden_dim), requires_grad=True)



    def reset(self,init_cell=None, batch_size = None):
        if init_cell:
            self._cell = init_cell
        else:
            if not batch_size:
                raise Exception(' Batch size needed')
            self._cell = torch.zeros(batch_size, self._num_kernels, self._hidden_dim)

    def get_forget(self, input, hidden):
        input_projection = input.squeeze(-1).transpose(0,1).matmul(self._input_to_forget).transpose(0,1) #batch_size X num_kernels X 1 X input_dimension * batch_size X num_kernels X input_dimension X projection_size-> batch_size Xnum_kernelsX 1 X projection_dimention
        hidden_projection = hidden.transpose(0,1).matmul(self._hidden_to_forget).transpose(0,1)
        forget = torch.nn.Sigmoid()(input_projection + hidden_projection + self._forget_bias.expand([input_projection.shape[0],-1,-1]))
        return forget

    def get_input(self, input, hidden):
        input_projection = input.squeeze(-1).transpose(0, 1).matmul(self._input_to_input).transpose(0,
                                                                                                     1)  # batch_size X num_kernels X 1 X input_dimension * batch_size X num_kernels X input_dimension X projection_size-> batch_size Xnum_kernelsX 1 X projection_dimention
        hidden_projection = hidden.transpose(0, 1).matmul(self._hidden_to_input).transpose(0, 1)
        input = torch.nn.Sigmoid()(
            input_projection + hidden_projection + self._input_bias.expand([input_projection.shape[0], -1, -1]))
        return input

    def get_output(self, input, hidden):
        input_projection = input.squeeze(-1).transpose(0, 1).matmul(self._input_to_out).transpose(0,
                                                                                                     1)  # batch_size X num_kernels X 1 X input_dimension * batch_size X num_kernels X input_dimension X projection_size-> batch_size Xnum_kernelsX 1 X projection_dimention
        hidden_projection = hidden.transpose(0, 1).matmul(self._hidden_to_out).transpose(0, 1)
        out = torch.nn.Sigmoid()(
            input_projection + hidden_projection + self._out_bias.expand([input_projection.shape[0], -1, -1]))
        return out


    def get_updated_cell(self, gated_forget, gated_input, input, hidden):
        forgotten_cell = gated_forget * self._cell
        input_projection = input.squeeze(-1).transpose(0, 1).matmul(self._input_to_cell).transpose(0,
                                                                                                     1)  # batch_size X num_kernels X 1 X input_dimension * batch_size X num_kernels X input_dimension X projection_size-> batch_size Xnum_kernelsX 1 X projection_dimention
        hidden_projection = hidden.transpose(0, 1).matmul(self._hidden_to_cell).transpose(0, 1)
        cell_projection = torch.nn.Tanh()(
            input_projection + hidden_projection + self._cell_bias.expand([input_projection.shape[0], -1, -1]))

        input_update = gated_input * torch.nn.Tanh()(cell_projection)
        self._cell = forgotten_cell + input_update
        return self._cell

    def get_updated_hidden(self, gated_output, updated_cell, hidden,mask):
        new_hidden = gated_output * torch.nn.Tanh()(updated_cell)
        mask = mask.map_(torch.zeros(mask.shape),lambda a,b:a>0).type(torch.BoolTensor)
        hidden.masked_scatter_(mask,new_hidden)
        return hidden

    def forward(self, hidden, input, mask):
        gated_forget = self.get_forget(input,hidden)
        gated_input = self.get_input(input, hidden)
        gated_output = self.get_output(input,hidden)
        updated_cell = self.get_updated_cell(gated_forget,gated_input, input, hidden)
        updated_hidden = self.get_updated_hidden(gated_output,updated_cell,hidden,mask)
        return updated_hidden



@Registrable.register('rim_module')
class RIMModule(torch.nn.Module,Registrable):

    def __init__(self, hidden_state_size, num_kernels, active_kernels, dynamics_module,input_dim, attended_input_dim):
        super().__init__()
        self._hidden_state_size = hidden_state_size
        self._num_kernels = num_kernels
        self._active_kernels = active_kernels
        self._dynamics_module = dynamics_module
        self._attended_input_dim = attended_input_dim
        self._input_dim = input_dim

        self._hidden_to_query_map = torch.nn.Parameter(torch.rand(num_kernels,hidden_state_size, attended_input_dim),requires_grad=True)
        self._input_to_key_map = torch.nn.Parameter(torch.rand(num_kernels,input_dim, attended_input_dim),requires_grad=True)
        self._input_to_values_map = torch.nn.Parameter(torch.rand(num_kernels,input_dim,attended_input_dim),requires_grad=True)

        self.reset_parameters()


    def reset_parameters(self):
        pass

    def reset(self,init_hidden=None, init_cell=None, batch_size =None):
        if init_hidden:
            self._rim_hidden_states = init_hidden
            self._dynamics_module.reset(init_cell)
        else:
            if not batch_size:
                raise Exception('batch size needed')
            self._rim_hidden_states = torch.zeros(batch_size, self._num_kernels, self._hidden_state_size)
            self._dynamics_module.reset(init_cell,batch_size)

    def get_update_mask(self, topk_indices):
        hidden_mask = torch.zeros(topk_indices.shape[0], self._num_kernels, self._hidden_state_size)

        #How to vectorize this ??
        for x in range(hidden_mask.shape[0]):
            hidden_mask[x].index_fill_(0,topk_indices[x],1)
        return  hidden_mask


    def get_attended_input(self, value, similarity_matrix,):
        '''

            #value: batch_size X num_filters X seq_len+1 X hidden_dim
            #similarity_matrix: batch_size X num_filters X seq_len +1
            #output: batch_sizeX num_filters X 1 X hidden_dim
        '''
        output = value.transpose(-1,-2).matmul(torch.nn.Softmax(dim=-1)(similarity_matrix).unsqueeze(-1))
        return output

    def update_from_hidden(self):
        raise NotImplementedError



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
        null_tensor = torch.zeros((input.shape[0],1,input.shape[2]), device=input.device)
        null_concated_input = torch.cat([input,null_tensor],dim=1)
        queries = self._rim_hidden_states.transpose(0,1).matmul(self._hidden_to_query_map).transpose(0,1) # batch_size X num_kernels X hidden_state_size * hidden_state_size X attended_input_dim -> num_kernels X attended_input_dim
        values = null_concated_input.unsqueeze(1).matmul(self._input_to_values_map) # batch_size X seq_len +1 X input_dim * num_kernels X input_dim X attended_input_dim  -> batch_size X num_kernels X seq_len + 1 X attended_input_dum
        keys = null_concated_input.unsqueeze(1).matmul(self._input_to_key_map) # batch_size X seq_len +1 X input_dim * num_kernels X input_dim X attended_input_dim  -> batch_size X num_kernels X seq_len + 1 X attended_input_dum
        attention_similarity_matrix = keys.matmul(queries.unsqueeze(-1)).squeeze(-1) # -> batch_size X num_kernsl X seq_len +1
        topk_vals, topk_indices = torch.topk(attention_similarity_matrix[:,:,-1], self._active_kernels,largest=False) # batch_size X num_kernels X 1

        #maybe remove the null dimension from the input?
        update_mask = self.get_update_mask(topk_indices)
        attended_input = self.get_attended_input(values,attention_similarity_matrix)
        updated_hidden = self._dynamics_module(self._rim_hidden_states, attended_input,update_mask)
        #updated_hidden += self._update_from_hidden()
        self._rim_hidden_states = updated_hidden
        return self._rim_hidden_states


@Seq2SeqEncoder.register('rim_encoder')
class RIMEncoder(Seq2SeqEncoder):
    pass

@SeqDecoder.register('rim_decoder')
class RIMDecoder(SeqDecoder):
    pass


if __name__ == '__main__':
    dynamics = LSTMDynamicsModule(32,128,3)
    rim_module = RIMModule(128,3,2, dynamics,64, 32)
    input = torch.rand(30,40,64)
    rim_module.reset(batch_size=30)
    rim_module(input)