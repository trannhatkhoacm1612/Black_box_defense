import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
def get_architech(arch):
    if arch == "unetsmall":
        return UNetSmall()
    elif arch == "cifar_encoder_192_24":
        model = Cifar_Encoder_192_24()
        return model
        
    elif arch == "cifar_decoder_192_24":
        model = Cifar_Decoder_192_24()
        return model
    
    elif arch == "dncnn":
        model = DnCNN(image_channels=3, depth=17, n_channels=64)
        return model







class encoding_block(nn.Module):
    """
    Convolutional batch norm block with relu activation (main block used in the encoding steps)
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=3,
        padding=0,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout=False,
    ):
        super().__init__()

        if batch_norm:

            # reflection padding for same size output as input (reflection padding has shown better results than zero padding)
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(out_size),
            ]

        else:
            layers = [
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    in_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
                nn.ReflectionPad2d(padding=(kernel_size - 1) // 2),
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.PReLU(),
            ]

        if dropout:
            layers.append(nn.Dropout())

        self.encoding_block = nn.Sequential(*layers)

    def forward(self, input):

        output = self.encoding_block(input)

        return output


# decoding block
class decoding_block(nn.Module):
    def __init__(self, in_size, out_size, batch_norm=False, upsampling=True):
        super().__init__()

        if upsampling:
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv = encoding_block(in_size, out_size, batch_norm=batch_norm)

    def forward(self, input1, input2):

        output2 = self.up(input2)

        output1 = nn.functional.upsample(input1, output2.size()[2:], mode="bilinear")

        return self.conv(torch.cat([output1, output2], 1))


class UNetSmall(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # encoding
        self.conv1 = encoding_block(3, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = encoding_block(32, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = encoding_block(64, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = encoding_block(128, 256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # center
        self.center = encoding_block(256, 512)

        # decoding
        self.decode4 = decoding_block(512, 256)
        self.decode3 = decoding_block(256, 128)
        self.decode2 = decoding_block(128, 64)
        self.decode1 = decoding_block(64, 32)

        # final
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, input):

        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # center
        center = self.center(maxpool4)

        # decoding
        decode4 = self.decode4(conv4, center)

        decode3 = self.decode3(conv3, decode4)

        decode2 = self.decode2(conv2, decode3)

        decode1 = self.decode1(conv1, decode2)

        # final
        final = nn.functional.upsample(
            self.final(decode1), input.size()[2:], mode="bilinear"
        )

        return final
    
    
# class Cifar_Encoder_192_24(nn.Module):
#     def __init__(self):
#         super(Cifar_Encoder_192_24, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 24, 4, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             # nn.ReLU(),
#  			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#              nn.ReLU(),
#             nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
#             nn.ReLU(),
# #         )
#     def forward(self, x):
#         encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
#         return encoded


# class Cifar_Decoder_192_24(nn.Module):
#     def __init__(self):
#         super(Cifar_Decoder_192_24, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
#             nn.ReLU(),
#              nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#              nn.ReLU(),
# 			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             # nn.ReLU(),
# 			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         decoded = self.decoder(x)
#         return decoded
    

class Cifar_Encoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192_24, self).__init__()
        # Input size: [batch, 3, 128, 128]
        # Output size: [batch, 192, 8, 8]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 192, 4, stride=2, padding=1),  # Thay đổi stride và padding ở đây
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)    # output size [batch, 192, 8, 8]
        return encoded

class Cifar_Decoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192_24, self).__init__()
        # Input size: [batch, 192, 16, 16]
        # Output size: [batch, 3, 128, 128]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class DnCNN(nn.Module):
    """
    This is a modified implementation of the DnCNN from https://github.com/cszn/DnCNN
    """
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        self.image_channels = image_channels
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        lastcnn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                lastcnn = m
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        init.constant_(lastcnn.weight, 0)