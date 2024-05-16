import torch
import torch.nn as nn
from torch.nn import Linear,ReLU,ModuleList,Sequential,Dropout,Softmax,Tanh
import torch.nn.functional as F
class MLP(torch.nn.Module):
    def __init__(self,input_dim=3,hidden_dim=32,hidden_layers=4,output_dim=3,activation=nn.ReLU()):
        super(MLP,self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input_layer', Linear(input_dim, hidden_dim))
        self.net.add_module('activation', activation)
        for i in range(hidden_layers):
            self.net.add_module(f'hidden_layer_{i}', Linear(hidden_dim, hidden_dim))
            self.net.add_module(f'activation_{i}', activation)
        self.net.add_module('output_layer', Linear(hidden_dim, output_dim))

    def forward(self,x):
        return self.net(x)

def gradients(u,x,order=1):
    if order==1:
        return torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True,only_inputs=True,)[0]
    else:
        return gradients(gradients(u,x,order-1),x,1)
default_criterion=nn.CrossEntropyLoss()
default_loss=torch.nn.MSELoss

