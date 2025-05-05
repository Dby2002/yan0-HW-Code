import torch.nn as nn


class HW1_Model(nn.Module):

    def __init__(self,infeature_len):
        super(HW1_Model,self).__init__()

        self.lever1 = nn.Sequential(
            nn.BatchNorm1d(infeature_len),
            nn.Linear(infeature_len,64),
            # nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Linear(64,16),
            # nn.BatchNorm1d(128),
            nn.Sigmoid(),
            # nn.Linear(32, 16),
            # nn.Sigmoid(),
            # nn.Linear(16,8),
            # nn.Sigmoid(),
            # nn.Linear(8,4),
            nn.Linear(16,1)
        )


    def forward(self,x):
        return self.lever1(x)