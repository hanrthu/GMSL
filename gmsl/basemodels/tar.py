# This is the implementation of task-aware readout block
import math
import torch
from torch_geometric.nn.inits import reset
import torch.nn as nn
from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_
from typing import Callable

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, input):
        u = input.mean(dim=-1, keepdim=True)
        s = (input - u).pow(2).mean(dim=-1, keepdim=True)
        output = (input - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * output + self.bias 

# Currently we can add multi-head cross attention mechanism to weight different nodes.
class TaskAwareReadout(nn.Module):
    def __init__(self, in_features, hidden_size, out_features,
        num_attention_heads: int = 8,
        dropout = 0.2,
        tasks = 6,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,):
        
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(TaskAwareReadout, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.tasks = tasks
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.query = nn.Linear(in_features, hidden_size)
        self.key = nn.Linear(in_features, hidden_size)
        self.value = nn.Linear(in_features, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        
        self.layernorm1 = LayerNorm(hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(hidden_size, eps=1e-12)
        self.linear1 = nn.Linear(in_features, 1)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def multi_head_reshape(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        # Shape (Batchsize, head_num, node_num, head_feature)
        return x.permute(0, 2, 1, 3)
    
    def reset_parameters(self):
        pass
        # # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        # self.weight_init(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     self.bias_init(self.bias)
        
    def forward(self, task_prompt, input):
        # Following the original Transformer structure, modified the attention mask
        # Prompt shape (1, task_num, hidden_feature)
        # Input shape (Batchsize, node_num, hidden_feature)
        query_full = self.query(task_prompt)
        key_full = self.key(input)
        value_full = self.value(input)
        
        query_multi = self.multi_head_reshape(query_full)
        key_multi = self.multi_head_reshape(key_full)
        value_multi = self.multi_head_reshape(value_full)
        
        attention_scores = torch.matmul(query_multi, key_multi.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        # mask = torch.ones((attention_scores.shape[-1], attention_scores.shape[-1]))
        # mask[0, 1:] = 0
        # Adding a mask that only the prompt can affect the nodes while the nodes cannot affect the prompts.
        # attention_scores = attention_scores * mask
        # Attention Probs shape (batchsize, head_num, task_num, node_num)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        context_layer = torch.matmul(attention_probs, value_multi)
        # Context shape (batchsize, task_num, head_num, head_feature)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size)
        # Context shape (batchsize, task_num, hidden_size)
        context_layer = context_layer.view(*new_shape)

        # We add the promt to influence the skip connections 
        context_layer = self.layernorm1(context_layer + self.linear1(task_prompt) * input)
        hidden_states = self.linear(context_layer)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm2(hidden_states + self.linear2(task_prompt) * context_layer)
        
        return hidden_states