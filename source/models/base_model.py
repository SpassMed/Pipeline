import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

def get_activation(atype):

    if atype=='relu':
        nonlinear = nn.ReLU()
    elif atype=='tanh':
        nonlinear = nn.Tanh()
    elif atype=='sigmoid':
        nonlinear = nn.Sigmoid() 
    elif atype=='elu':
        nonlinear = nn.ELU()
    return nonlinear

def make_dense_layer(in_dim, out_dim, atype: str='relu'):
    
    layer = nn.Linear(in_dim, out_dim)
    bn = nn.BatchNorm1d(out_dim)
    
    if atype:
        nonlinear = get_activation(atype)
        return [layer, bn, nonlinear]
    
    else: return [layer]

def make_lazy_dense_layer(out_dim, atype: str='relu'):
    
    layer = nn.LazyLinear(out_dim)
    bn = nn.BatchNorm1d(out_dim)
    
    if atype:
        nonlinear = get_activation(atype)
        return nn.Sequential(*[layer, bn, nonlinear])
    
    else: return nn.Sequential(*[layer])

def makeblock_conv(in_chs:int, out_chs:int, kernel_size:int=5, atype:str='relu', stride:int=1):

    layer = nn.Conv1d(in_channels=in_chs, 
        out_channels=out_chs, kernel_size=kernel_size, stride=stride, padding='same')
    bn = nn.BatchNorm1d(out_chs)
    nonlinear = get_activation(atype)
    mx_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    return nn.Sequential(*[layer, bn, nonlinear, mx_pool])


class MakeBlockConv(nn.Module):

    def __init__(self, in_chs:int, out_chs:int, kernel_size:int=5, atype:str='relu', stride:int=1):
        super().__init__()

        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.act_type = atype
        self.stride = stride

        self.layer = nn.Conv1d(in_channels=self.in_chs, 
            out_channels=self.out_chs, kernel_size=self.kernel_size, stride=self.stride, padding='same')
        self.bn_layer = nn.BatchNorm1d(self.out_chs)
        self.nonlinear = get_activation(self.act_type)
        self.mx_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.layer(x)
        x = self.bn_layer(x)
        x = self.nonlinear(x)
        x = self.mx_pool(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    # return nn.Sequential(*[layer, bn, nonlinear, mx_pool])

def makevggblock_conv(in_chs:int, out_chs:int, kernel_size:int=5, atype:str='relu', stride:int=1, isBN:bool=False):

    layer_1 = nn.Conv1d(in_channels=in_chs, 
        out_channels=out_chs, kernel_size=kernel_size, stride=stride, padding='same')
    bn_1 = nn.BatchNorm1d(out_chs)
    nonlinear_1 = get_activation(atype)
    
    layer_2 = nn.Conv1d(in_channels=out_chs, 
        out_channels=out_chs, kernel_size=kernel_size, stride=stride, padding='same')
    bn_2 = nn.BatchNorm1d(out_chs)
    nonlinear_2 = get_activation(atype)

    mx_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    if isBN:
        return nn.Sequential(*[layer_1, bn_1, nonlinear_1, layer_2, bn_2, nonlinear_2, mx_pool])
    else: 
        return nn.Sequential(*[layer_1, nonlinear_1, layer_2, nonlinear_2, mx_pool])


### MLP ###
class ModelVanillaMLP(nn.Module):

    def __init__(self, input_size:int, output_size:int, hidden_width: int, n_hidden_layers: int, **kwargs):
        super(ModelVanillaMLP, self).__init__()

        # add input layer
        self.input_size = input_size
        self.output_size = output_size
        if isinstance(hidden_width, int):
            self.first_hidden_output = hidden_width
            self.hidden_width = hidden_width
            
        # To do # implement different hidden widths for hidden layers    
        elif isinstance(hidden_width, list): 
            self.first_hidden_output = hidden_width[0]
            self.hidden_width = hidden_width

        model_list = [nn.Linear(input_size, self.first_hidden_output), nn.ReLU()]

        for _ in range(n_hidden_layers):
            model_list.extend(make_dense_layer(hidden_width, hidden_width))

        model_list.extend(make_dense_layer(hidden_width, output_size, atype=None))
        self.sequential = nn.Sequential(*model_list)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = x.view(-1, self.input_size)
        return self.sequential(x)
    
