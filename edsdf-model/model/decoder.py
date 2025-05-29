import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

def activate(act, inplace=False, neg_slope=0.2):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        se_weight = self.se(x)
        return x * se_weight

class MainPart(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6'):
        super(MainPart, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                activate(self.activation, inplace=True),
                
            )
            for kernel_size in self.kernel_sizes
        ])
        

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)

        result = outputs[0]
        for output in outputs[1:]:
            result = result * output

        return result
    

class EMSFBlock(nn.Module):# Efficient Multi-Scale Fusion Block
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5,7], expansion_factor=2, activation='relu6'):
        super(EMSFBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.activation = activation
        
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_channels = int(self.in_channels * self.expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.ex_channels),
            activate(self.activation, inplace=True)
        )

        self.MainPart = MainPart(self.ex_channels, self.kernel_sizes, self.stride, self.activation)

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.ex_channels, self.out_channels, 1, 1, 0, bias=False, groups=self.out_channels),
            nn.BatchNorm2d(self.out_channels),
        )

        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        MainPart_outs = self.MainPart(pout1)
        dout = MainPart_outs
        dout = channel_shuffle(dout, gcd(self.ex_channels,self.out_channels))
        out = self.pconv2(dout)
        
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def MainDecoder(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5, 7], expansion_factor=2, activation='relu6'):
    convs = []
    
    main_module = EMSFBlock(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation)
    convs.append(main_module)
    
    if n > 1:
        for i in range(1, n):
            a_module = EMSFBlock(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, activation=activation)
            convs.append(a_module)
    
    conv = nn.Sequential(*convs)
    
    return conv

class EFUB(nn.Module): # Efficient Fusion Upsampling Block
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], stride=1, activation='relu'):
        super(EFUB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.up_dwc = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                activate(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])

        # Pointwise Convolution
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = [dwconv(x) for dwconv in self.up_dwc]
        
        x = outputs[0]
        for output in outputs[1:]:
            x = x * output
        
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        
        return x


class MSGAG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5], groups=1):
        super(MSGAG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
                nn.BatchNorm2d(out_channels)
            )
            for kernel_size in kernel_sizes
        ])

        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = [group_conv(x) for group_conv in self.group_convs]
        x = outputs[0]
        for output in outputs[1:]:
            x = x * output

        psi = self.psi(x)

        return x * psi
    

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y
    

def get_dct_filter(tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
    dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

    for t_x in range(tile_size_x):
        for t_y in range(tile_size_y):
            dct_filter[:, t_x, t_y] = build_filter(t_x, mapper_x, tile_size_x) * build_filter(t_y, mapper_y, tile_size_y)

    return dct_filter

def build_filter(pos, freq, POS):
    result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

class DCTHighFrequencyExtractor(nn.Module):
    def __init__(self, in_channels, dct_h, dct_w, frequency_branches=16, frequency_selection='top'):
        super(DCTHighFrequencyExtractor, self).__init__()

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection + str(frequency_branches))
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature = 0
        for name, params in self.named_buffers():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature += F.adaptive_avg_pool2d(x_pooled_spectral, (H, W))

        return multi_spectral_feature / self.num_freq
    
#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = activate(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class EFBEB(nn.Module): # Efficient Frequency Boundary Extraction Block
    def __init__(self, in_channels, frequency_branches=16, frequency_selection='top', activation='relu'):
        super(EFBEB, self).__init__()

        c2wh = dict([(32, 112), (64, 56), (128, 28), (256, 14), (512, 7), (320, 12), (160, 20)])
        assert in_channels in c2wh, "The input channels must be one of [32, 64, 128, 256, 512, 320, 160]"

        dct_h = c2wh[in_channels]
        dct_w = c2wh[in_channels]

        self.dct_extractor = DCTHighFrequencyExtractor(in_channels, dct_h, dct_w, frequency_branches, frequency_selection)

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.ReLU6(inplace=True)

        self.cab = CAB(in_channels, activation=activation)
        self.sab = SAB()

        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        high_freq_features = self.dct_extractor(x)
        out1 = self.dwconv(high_freq_features)
        out2 = self.bn(out1)
        out3 = self.activation(out2)

        cab_out = self.cab(out3)
        cab_out = out3 * cab_out
        sab_out = self.sab(out3)
        sab_out = out3 * sab_out
        
        attention_out = cab_out * sab_out

        output = attention_out + x
        output = self.conv1x1(output)
        
        return output



class EDSD(nn.Module): # Efficient Deep Supervision Decoder
    def __init__(self, channels=[256,160,64,32], kernel_sizes1=[1,3,5,7], kernel_sizes2=[1,3,5,7], expansion_factor=2, activation='relu6'):
        super(EDSD,self).__init__()
        
        up_ks = [3,3]# ks for up block
        self.mainblock4 = MainDecoder(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes1, expansion_factor=expansion_factor, activation=activation)
        self.assist4 = EFBEB(in_channels=channels[0], frequency_branches=16, frequency_selection='top', activation=activation )

        self.up3 = EFUB(in_channels=channels[0], out_channels=channels[1], kernel_sizes=up_ks, stride=1, activation=activation)
        self.msgag3 = MSGAG(in_channels=channels[1], out_channels=channels[1], kernel_sizes=kernel_sizes2, groups=channels[1]//4)

        self.mainblock3 = MainDecoder(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes1, expansion_factor=expansion_factor, activation=activation)
        self.assist3 = EFBEB(in_channels=channels[1], frequency_branches=16, frequency_selection='top', activation=activation )

        self.up2 = EFUB(in_channels=channels[1], out_channels=channels[2], kernel_sizes=up_ks, stride=1, activation=activation)
        self.msgag2 = MSGAG(in_channels=channels[2], out_channels=channels[2], kernel_sizes=kernel_sizes2, groups=channels[2]//4)
        
        self.mainblock2 = MainDecoder(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes1, expansion_factor=expansion_factor, activation=activation)
        self.assist2 = EFBEB(in_channels=channels[2], frequency_branches=16, frequency_selection='top', activation=activation )

        self.up1 = EFUB(in_channels=channels[2], out_channels=channels[3], kernel_sizes=up_ks, stride=1, activation=activation)
        self.msgag1 = MSGAG(in_channels=channels[3], out_channels=channels[3], kernel_sizes=kernel_sizes2, groups=int(channels[3]/4))
        
        self.mainblock1 = MainDecoder(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes1, expansion_factor=expansion_factor, activation=activation)
        self.assist1 = EFBEB(in_channels=channels[3], frequency_branches=16, frequency_selection='top', activation=activation )    

      
    def forward(self, x, skips):
        
        #4
        d4 = self.mainblock4(x)
        boundary4 = self.assist4(d4)

        #up3,msgag3
        d3 = self.up3(d4)
        x3 = skips[0]
        y3 = d3 + x3
        y3 = self.msgag3(y3)

        #Addition
        d3 = d3 + y3
        
        #3
        d3 = self.mainblock3(d3)
        boundary3 = self.assist3(d3)

        #up2,msgag2
        d2 = self.up2(d3)
        x2 = skips[1]
        y2 = d2 + x2
        y2 = self.msgag2(y2)

        #Addition
        d2 = d2 + y2

        #2
        d2 = self.mainblock2(d2)
        boundary2 = self.assist2(d2)

        #up1,msgag1
        d1 = self.up1(d2)
        x1 = skips[2]
        y1 = d1 + x1
        y1 = self.msgag1(y1)

        #Addition
        d1 = d1 + y1

        #1
        d1 = self.mainblock1(d1)
        boundary1 = self.assist1(d1)
        

        
        return [d4, d3, d2, d1], [boundary4, boundary3, boundary2, boundary1]
    
class EDSD1(nn.Module):  # Efficient Deep Supervision Decoder
    def __init__(self, channels=[256, 160, 64, 32], kernel_sizes1=[1, 3, 5, 7], kernel_sizes2=[1, 3, 5, 7], 
                 expansion_factor=2, use_se=True, activation='relu6'):
        super(EDSD1, self).__init__()

        up_ks = [3, 3]

        self.mainblocks = nn.ModuleList()
        self.assists = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.msgags = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):  
            self.mainblocks.append(
                MainDecoder(channels[i], channels[i], n=1, stride=1, kernel_sizes=kernel_sizes1, 
                            expansion_factor=expansion_factor, 
                            use_se=use_se, activation=activation)
            )
            self.assists.append(
                EFBEB(in_channels=channels[i], frequency_branches=16, frequency_selection='top', activation=activation)
            )
            self.upblocks.append(
                EFUB(in_channels=channels[i], out_channels=channels[i-1], kernel_sizes=up_ks, stride=1, activation=activation)
            )
            self.msgags.append(
                MSGAG(in_channels=channels[i-1], out_channels=channels[i-1], kernel_sizes=kernel_sizes2, 
                      groups=channels[i-1]//4)
            )
        
    def forward(self, x, skips):
        outputs = []
        boundaries = []


        for i in range(len(self.mainblocks)):

            d = self.mainblocks[i](x)
            boundary = self.assists[i](d)

            d = self.upblocks[i](d)
            skip = skips[i]
            y = d + skip
            y = self.msgags[i](y)

            d = d + y
            outputs.append(d)
            boundaries.append(boundary)

            x = d

        return outputs, boundaries
