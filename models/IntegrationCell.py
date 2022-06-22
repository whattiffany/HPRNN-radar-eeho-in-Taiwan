import torch
import torch.nn as nn

class IntegrationCell(nn.Module):
    def __init__(self):
        super(IntegrationCell,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                    out_channels=64, 
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, 
                    out_channels=128, 
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x_out = torch.tensor([]).cuda()
        for i in range(x.size(1)):
            new_x = self.conv(x[:,i,:,:,:])
            x_out = torch.cat([x_out,new_x.unsqueeze(1)],1)
            # print(x_out.size())
        return x_out

class GuidingCell(nn.Module):
    def __init__(self):
        super(GuidingCell,self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, 
                    out_channels=64, 
                    kernel_size=3,
                    stride=1,
                    padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, 
                    out_channels=64, 
                    kernel_size=3,
                    stride=1,padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x_out = torch.tensor([]).cuda()
        for i in range(x.size(0)):
            new_x = self.deconv(x[i,:,:,:,:])
            x_out = torch.cat([x_out,new_x.unsqueeze(0)],0)
            # print("x_out",x_out.size())
        return x_out