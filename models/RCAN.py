import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size-1)//2,  # same padding
        bias=bias)


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                conv(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                conv(channel // reduction, channel, 1),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, args):
        super(RCAB, self).__init__()
        # init
        self.num_feat = args.num_feat
        self.reduction = args.reduction

        # body
        modules_body = [
            conv(self.num_feat, self.num_feat),
            nn.ReLU(True),
            conv(self.num_feat, self.num_feat),
            CALayer(self.num_feat, self.reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, args):
        super(ResidualGroup, self).__init__()
        # init
        self.n_resblocks = args.n_resblocks
        self.num_feat = args.num_feat

        # body
        modules_body = [RCAB(args) for _ in range(self.n_resblocks)]
        modules_body.append(conv(self.num_feat, self.num_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args):
        super(RCAN, self).__init__()
        # init
        self.num_channel = args.num_channel
        self.num_feat = args.num_feat
        self.scale = args.scale
        self.n_resgroups = args.n_resgroups

        # shallow
        self.shallow = conv(self.num_channel, self.num_feat)

        # RIR
        modules_body = [ResidualGroup(args) for _ in range(self.n_resgroups)]
        modules_body.append(conv(self.num_feat, self.num_feat))
        self.body = nn.Sequential(*modules_body)

        # Upsampler
        self.conv_up = conv(self.num_feat, self.num_feat*self.scale*self.scale)
        self.upsample = nn.PixelShuffle(self.scale) 
        self.conv_out = conv(self.num_feat, self.num_channel)

    def forward(self, x):
        # shallow feature
        x = self.shallow(x)

        # RIR
        res = self.body(x)

        # Residual
        res += x

        # upsample
        up = self.conv_up(res)
        up = self.upsample(up) 
        x = self.conv_out(up)

        return x