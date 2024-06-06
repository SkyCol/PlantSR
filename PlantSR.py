import torch.nn as nn
import torch

def conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size-1)//2,  # same padding
        bias=bias)

# SE ccchannel Attention Layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,h,w = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)


# Residual Channel Attention Block (RCAB)
class RSEB(nn.Module):
    def __init__(self, num_feat,reduction):
        super(RSEB, self).__init__()
        # init
        self.num_feat = num_feat
        self.reduction = reduction

        # body
        modules_body = [
            conv(self.num_feat, self.num_feat),
            nn.ReLU(True),
            conv(self.num_feat, self.num_feat),
            SELayer(self.num_feat, self.reduction)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_resblocks,num_features,reduction):
        super(ResidualGroup, self).__init__()
        # init
        self.n_resblocks = n_resblocks
        self.num_feat = num_features
        self.reduction = reduction

        # body
        modules_body = [RSEB(self.num_feat,self.reduction) for _ in range(self.n_resblocks)]
        modules_body.append(conv(self.num_feat, self.num_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Plant Super Resolution Network (PlantSR)
class PlantSR(nn.Module):
    def __init__(self, scale, num_channels=3, num_features=64,n_resgroups=16,n_resblocks=4,reduction=16):
        super(PlantSR, self).__init__()
        # init
        self.num_channel = num_channels


        self.num_features = num_features
        self.scale = scale
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.reduction = reduction
        
        
        # shallow feature extraction
        self.shallow = conv(self.num_channel, self.num_features)

        # middle feature extraction
        self.middle_layers = self.n_resgroups // 2
        self.deep_layers = self.n_resgroups-self.middle_layers
        modules_middle = [ResidualGroup(self.n_resblocks,self.num_features,self.reduction) for _ in range(self.middle_layers)]
        modules_middle.append(conv(self.num_features, self.num_features))
        self.middle = nn.Sequential(*modules_middle)

        # deep feature extraction
        modules_deep = [ResidualGroup(self.n_resblocks,self.num_features,self.reduction) for _ in range(self.deep_layers)]
        modules_deep.append(conv(self.num_features, self.num_features))
        self.deep = nn.Sequential(*modules_deep)
        

        # Upsampler
        self.conv_up = conv(self.num_features, self.num_features*self.scale*self.scale)
        self.upsample = nn.PixelShuffle(self.scale)
        self.conv_out = conv(self.num_features, self.num_channel)

    def forward(self, x):
        
        # shallow feature
        x = self.shallow(x)

        # Middle feature
        m = self.middle(x)

        # deep feature
        d = self.deep(m)

        # Residual
        d+=x
        d+=m

        # upsample
        up = self.conv_up(d)
        up = self.upsample(up) 
        x = self.conv_out(up)
        
        return x
