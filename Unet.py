from torch.nn import Module, Sequential, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, Dropout3d, ReLU, Sigmoid
import torch
from torch.autograd import Variable
from torchsummaryX import summary

class UNet(Module):
    def __init__(self, 
                channels = 1, 
                residual = 'conv'):

        super(UNet, self).__init__()

        channel_64 = 64
        channel_128 = 128
        channel_256 = 256
        channel_512 = 512
        channel_1024 = 1024

        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        self.conv_blok_1 = Conv3DBlock(channels, channel_64, residual = residual)
        self.conv_blok_2 = Conv3DBlock(channel_64, channel_128, residual = residual)
        self.conv_blok_3 = Conv3DBlock(channel_128, channel_256, residual = residual)
        self.conv_blok_4 = Conv3DBlock(channel_256, channel_512, residual = residual)
        self.conv_blok_5 = Conv3DBlock(channel_512, channel_1024, residual = residual)

        self.decoder_conv_blok_4 = Conv3DBlock(2 * channel_512, channel_512, residual = residual)
        self.decoder_conv_blok_3 = Conv3DBlock(2 * channel_256, channel_256, residual = residual)
        self.decoder_conv_blok_2 = Conv3DBlock(2 * channel_128, channel_128, residual = residual)
        self.decoder_conv_blok_1 = Conv3DBlock(2 * channel_64, channel_64, residual = residual)

        self.deconv_blok_4 = Deconv3DBlock(channel_1024, channel_512)
        self.deconv_blok_3 = Deconv3DBlock(channel_512, channel_256)
        self.deconv_blok_2 = Deconv3DBlock(channel_256, channel_128)
        self.deconv_blok_1 = Deconv3DBlock(channel_128, channel_64)

        self.final_conv = Conv3d(channel_64, channels, kernel_size = 1, stride = 1, padding = 0, bias = True)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encoder 
        conv1 = self.conv_blok_1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv_blok_2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv_blok_3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv_blok_4(pool3)

        pool4 = self.pool4(conv4)
        conv_middle = self.conv_blok_5(pool4)

        # Decoder 
        deconv4 = torch.cat([self.deconv_blok_4(conv_middle), conv4], dim=1)
        uconv4 = self.decoder_conv_blok_4(deconv4)

        deconv3 = torch.cat([self.deconv_blok_3(uconv4), conv3], dim=1)
        uconv3 = self.decoder_conv_blok_3(deconv3)
        uconv3 = Dropout3d(p = 0.5)(uconv3)

        deconv2 = torch.cat([self.deconv_blok_2(uconv3), conv2], dim=1)
        uconv2 = self.decoder_conv_blok_2(deconv2)
        uconv2 = Dropout3d(p = 0.5)(uconv2)

        deconv1 = torch.cat([self.deconv_blok_1(uconv2), conv1], dim=1)
        uconv1 = self.decoder_conv_blok_1(deconv1)

        output_layer  = self.sigmoid(self.final_conv(uconv1))

        return output_layer 

class Conv3DBlock(Module):
    def __init__(self, 
                input_channels, 
                output_channels, 
                kernel_size = 3, 
                stride = 1, 
                padding = 1, 
                residual = None):

        super(Conv3DBlock, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(input_channels, 
                                output_channels, 
                                kernel_size = kernel_size,
                                stride = stride, 
                                padding = padding, 
                                bias = True),
                        BatchNorm3d(output_channels),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(output_channels, 
                                output_channels, 
                                kernel_size = kernel_size,
                                stride = stride, 
                                padding = padding, 
                                bias = True),
                        BatchNorm3d(output_channels),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(input_channels, 
                                            output_channels, 
                                            kernel_size = 1, 
                                            bias = False)

    def forward(self, x):
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv3DBlock(Module):
    def __init__(self, 
                input_channels, 
                output_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1):
                
        super(Deconv3DBlock, self).__init__()

        self.deconv = Sequential(
                        ConvTranspose3d(input_channels, 
                                        output_channels, 
                                        kernel_size = (kernel_size, kernel_size, kernel_size),
                                        stride = (stride, stride, stride), 
                                        padding = (padding, padding, padding), 
                                        output_padding = 1, 
                                        bias = True),
                        ReLU())

    def forward(self, x):
        return self.deconv(x)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    net =UNet(residual='pool').cuda().eval()
    data = Variable(torch.randn(1, 1, 32, 32, 32)).cuda()
    out = net(data)
    summary(net, data)
    print("out size: {}".format(out.size()))