# This is the implementation of task-aware readout block
import math
import torch
from torch_geometric.nn.inits import reset
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_
from typing import Callable


class MultiLayerTAR(nn.Module):
    def __init__(self, in_features, hidden_size, out_features,
        num_attention_heads: int = 8,
        dropout = 0.2,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
        num_layers = 1):
        super(MultiLayerTAR, self).__init__()
        self.num_layers = num_layers
        self.readout = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.readout.append(TaskAwareReadout(in_features=in_features, hidden_size=hidden_size, out_features=out_features, num_attention_heads=num_attention_heads, 
                                                       dropout=dropout, weight_init=weight_init, bias_init=bias_init))
            else:
                self.readout.append(TaskAwareReadout(in_features=out_features, hidden_size=hidden_size, out_features=out_features, num_attention_heads=num_attention_heads, 
                                                       dropout=dropout, weight_init=weight_init, bias_init=bias_init))
            
     
    def forward(self, task_prompt, input, index):
        hidden_states = task_prompt
        for i in range(self.num_layers):
             hidden_states = self.readout[i](hidden_states, input, index)
        # output_feature = hidden_states.permute(1, 0, 2)
        return hidden_states
    
class TaskAwareReadout(nn.Module):
    def __init__(self, in_features, hidden_size, out_features,
        num_attention_heads: int = 8,
        dropout = 0.2,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,):
        
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(TaskAwareReadout, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(in_features, hidden_size)
        self.key = nn.Linear(in_features, hidden_size)
        self.value = nn.Linear(in_features, hidden_size)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.linear1 = nn.Linear(in_features, 1)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def multi_head_reshape(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        # Shape (Batchsize, head_num, node_num, head_feature)
        return x.permute(0, 2, 1, 3)
    
    # def reset_parameters(self):
    #     # # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
    #     self.weight_init(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         self.bias_init(self.bias)
    

    def padded_batch_generation(self, input, index):
        # Process the input to get a padded matrix
        unique_index, counts = torch.unique(index, return_counts=True)
        max_len = torch.max(counts).item()
        padded_tensors = []
        for item in unique_index:
            unique_tensor = input[index==item, :]
            if len(unique_tensor) < max_len:
                padded = torch.zeros((max_len-unique_tensor.shape[0], unique_tensor.shape[1])).to(unique_tensor.device)
                unique_tensor = torch.cat([unique_tensor, padded], dim=0)
            padded_tensors.append(unique_tensor)
        batched_input = torch.stack(padded_tensors, dim=0)
        return batched_input
    
    def forward(self, task_prompt, input, index):
        # This is the task aware attention readout
        # Following the original Transformer structure, modified the attention mask
        # Prompt shape (1, task_num, hidden_feature)
        # Original input shape (node_num, hidden_feature)
        # Processed input shape (Batchsize or Chainsize, node_num, hidden_feature)
        input = self.padded_batch_generation(input, index)
        query_full = self.query(task_prompt)
        key_full = self.key(input)
        value_full = self.value(input)
        
        query_multi = self.multi_head_reshape(query_full)
        key_multi = self.multi_head_reshape(key_full)
        value_multi = self.multi_head_reshape(value_full)
        # print("Query Shape:", query_multi.shape)
        # print("Key Shape:", key_multi.shape)
        attention_scores = torch.matmul(query_multi, key_multi.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        
        # mask = torch.ones((attention_scores.shape[-1], attention_scores.shape[-1]))
        # mask[0, 1:] = 0
        # print("Attention Shape:", attention_scores.shape, mask.shape)
        # Adding a mask that only the prompt can affect the nodes while the nodes cannot affect the prompts.
        # attention_scores = attention_scores * mask
        # Attention Probs shape (batchsize, head_num, task_num, node_num)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        context_layer = torch.matmul(attention_probs, value_multi)
        # Context shape (batchsize, task_num, head_num, head_feature)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size, )
        # Context shape (batchsize, task_num, hidden_size)
        context_layer = context_layer.view(*new_shape)

        # We add the promt to influence the skip connections 
        # TODO: 这里是否加query_full有待验证
        context_layer = self.layernorm1(context_layer + self.linear1(task_prompt) * query_full)
        hidden_states = self.linear(context_layer)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm2(hidden_states + self.linear2(task_prompt) * context_layer)
        # task_num, batchsize(or chainsize), hidden_size
        hidden_states = hidden_states.permute(1, 0, 2) # [task_num, batchsize(or chainsize), hidden_size]
        return hidden_states