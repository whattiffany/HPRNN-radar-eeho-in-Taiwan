from numpy import pad
import torch.nn as nn

class EncoderCNNCell(nn.Module):
    def __init__(self):
        super(EncoderCNNCell,self).__init__()
        """
        Initialize Encoder_CNN cell.
        Parameters
        ----------
        in_channel: int
            Number of channels of input tensor.
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                    out_channels=16, 
                    kernel_size=5,
                    stride=1,
                    padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, 
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, 
                    out_channels=64, 
                    kernel_size=3,
                    stride=2,
                    padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv(x)
        # print("conv=", x.size())
        return x

class DecoderCNNCell(nn.Module):
    def __init__(self):
        super(DecoderCNNCell,self).__init__()
        """
        Initialize Decoder_CNN cell.
        Parameters
        ----------
        in_channel: int
            Number of channels of input tensor.
        """

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, 
                    out_channels=32, 
                    kernel_size=2,
                    stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, 
                    out_channels=16,
                    kernel_size=2,
                    stride=2,
                    ),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, 
                    out_channels=1, 
                    kernel_size=5,
                    stride=1,
                    padding=2),
            nn.ReLU()
        )
        
    def forward(self, x):
        # print('de conv')    
        x = self.deconv(x)
        # print("de conv=", x.size())
        return x