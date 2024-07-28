from torch import dropout
import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, in_dim=0, out_dim=0, hidden_list = []):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(p=drop))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
       
        return self.layers(x)

     
class Linear_CNN(nn.Module):
    def __init__(
            self, in_dim=0, out_dim=0, hidden_list = [],
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Conv3d(lastv, hidden, kernel_size=1, bias=bias))
            if bn:
                layers.append(nn.BatchNorm3d(hidden))
            layers.append(act)
            lastv = hidden
        layers.append(nn.Conv3d(lastv, out_dim, kernel_size=1, bias=bias))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)