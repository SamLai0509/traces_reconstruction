"""
Model for using the Resnet Encoder-Decoder Model for the traces reconstructions.

"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    encoder: 2 1-d convolution layers in one block. There are 3 blocks in the encoder.

    decoder: 2 1-d convotranspose layers in one block.  There are 3 blocks in the decoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding):
        super(DecoderResidualBlock, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)

        # Adjust channels in skip connection if necessary
        self.adjust_channels = nn.ConvTranspose1d(in_channels, out_channels, 1, stride=2, output_padding=output_padding) if in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply the skip connection
        if self.adjust_channels is not None:
            identity = self.adjust_channels(identity)

        out += identity
        out = self.relu(out)
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, input_size=1024, kernel_size=3):
        super(Autoencoder, self).__init__()
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        padding = kernel_size // 2

        # Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            ResidualBlock(3, 32, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(32, 64, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2),
            ResidualBlock(64, 128, kernel_size, stride=1, padding=padding),
            nn.MaxPool1d(2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            DecoderResidualBlock(128, 64, kernel_size, padding=kernel_size//2, output_padding=1),
            DecoderResidualBlock(64, 32, kernel_size, padding=kernel_size//2, output_padding=1),
            nn.ConvTranspose1d(32, 3, kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
    
    
