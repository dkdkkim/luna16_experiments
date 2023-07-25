'''
- reproduction from Hang Zhang, et al. "ResNeSt: Split-Attention Networks" 
- reference paper: https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf
- reference code: https://github.com/STomoya/ResNeSt
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RSoftmax(nn.Module):
    def __init__(self, r, cardinality) -> None:
        super(RSoftmax, self).__init__()
        self.r = r
        self.cardinality = cardinality
    
    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.cardinality, self.r, -1).transpose(1,2)
        x = F.softmax(x, dim=1)
        x = x.reshape(batch, -1, 1, 1, 1)
        return x
        
class SplitAttention(nn.Module):
    def __init__(self,
                 in_ch,
                 mid_ch,
                 r,
                 cardinality) -> None:
        super(SplitAttention, self).__init__()
        self.in_ch = in_ch
        self.r = r
        self.fc1 = nn.Conv3d(in_ch//r, mid_ch, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Conv3d(mid_ch, in_ch, 1, groups=cardinality)
        self.bn2 = nn.BatchNorm3d(in_ch)
        self.rsoftmax = RSoftmax(r, cardinality)
    
    def forward(self, x):
        '''
        split : (c * k) * r     [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]
        '''
        # print(x.size(), self.in_ch//self.r, self.in_ch, self.r)
        splited = torch.split(x, self.in_ch//self.r, dim=1)
        # print('split', splited[0].size())
        '''
        sum   : c * k         | group 0 | group 1 | ...| group k |
        '''
        gap = sum(splited)
        gap = F.adaptive_avg_pool3d(gap, 1)
        # print(gap.size())
        '''
        fc1 : c' * k   
        fc2 : c  * r * k
        '''
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        gap = self.fc2(gap)
        gap = self.bn2(gap)
        # print(gap.size())
        attention = self.rsoftmax(gap)
        # print(gap.size())
        '''
        attention split : c * k * r
        '''
        attention = torch.split(attention, self.in_ch//self.r, dim=1)
        # print('attention', attention[0].size())
        out = sum([a*s for (a, s) in zip(attention, splited)])
        
        return out.contiguous() # To prevent 'RuntimeError: input is not contiguous'

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self,
                in_ch,
                bottleneck_width,
                kernel_size,
                stride,
                padding,
                dilation,
                split_r,
                reduction=4,
                cardinality=32,
                downsample=None
                ) -> None:
        super(BottleneckBlock, self).__init__()
    
        self.conv1 = nn.Conv3d(in_ch, bottleneck_width, 1, 1, 0, dilation=dilation, groups=cardinality*split_r, bias=False)
        self.bn1 = nn.BatchNorm3d(bottleneck_width)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(bottleneck_width, bottleneck_width*split_r, kernel_size, stride, padding, dilation, cardinality*split_r, bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_width*split_r)
        
        self.split_attention = SplitAttention(bottleneck_width*split_r, bottleneck_width*split_r//reduction, split_r, cardinality)
        
        self.conv3= nn.Conv3d(bottleneck_width, bottleneck_width*4, 1, 1, 0, dilation=dilation, groups=cardinality, bias=False)
        self.bn3 = nn.BatchNorm3d(bottleneck_width*4)
        self.downsample = downsample
            
    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.split_attention(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNeSt(nn.Module):
    def __init__(self,
                 block,
                 layer_nums,
                 init_width,
                 block_width,
                 num_classes,
                 in_ch,
                 r=2,
                 init_down=False,
                 deep_stem=False) -> None:
        super(ResNeSt, self).__init__()
        
        if init_down:
            init_stride = 2
        else:
            init_stride =1
        
        if deep_stem:
            self.init_conv = nn.Sequential(
                nn.Conv3d(in_ch, int(init_width/2), kernel_size=3, stride=init_stride, padding=1, bias=False),
                nn.BatchNorm3d(int(init_width/2)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(int(init_width/2), int(init_width/2), kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(int(init_width/2)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(int(init_width/2), init_width, kernel_size=3, stride=1, padding=1, bias=False)
            )
        else:
            self.init_conv = nn.Conv3d(in_ch, init_width, kernel_size=5, stride=init_stride, padding=2, bias=False)

        self.bn1 = nn.BatchNorm3d(init_width)
        self.relu = nn.ReLU(True)
        
        self.in_plane = init_width
        self.block1 = self._make_layer(block, block_width, layer_nums[0], r, False)
        self.block2 = self._make_layer(block, block_width * 2, layer_nums[1], r, True)
        self.block3 = self._make_layer(block, block_width * 4, layer_nums[2], r, True)
        self.block4 = self._make_layer(block, block_width * 8, layer_nums[3], r, True)
        
        self.globalavgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(block_width * 8 * block.expansion, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_normal_(m.weight.data, gain=1.0)
    
    def _make_layer(self,
                      block,
                      block_width,
                      n_blocks,
                      r,
                      is_downsample):
        layer, down_layer = [], []
        down_layer.append(nn.Conv3d(self.in_plane, block_width * block.expansion, 1, 1, bias=False))
        if is_downsample:
            down_layer.append(nn.AvgPool3d(2))
            layer.append(
                block(self.in_plane, block_width, 3, 2, 1, 1, r, downsample=nn.Sequential(*down_layer))
            )
        else:
            layer.append(
                block(self.in_plane, block_width, 3, 1, 1, 1, r, downsample=nn.Sequential(*down_layer))
            )
        self.in_plane = block_width * block.expansion
        for i in range(1, n_blocks):
            layer.append(
                block(self.in_plane, block_width, 3, 1, 1, 1, r)
            )
        return nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
                
        x = self.globalavgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
        
        
def generate_model(model_depth, init_width=64, block_width=128, in_ch=1, n_class=2, r=2, init_down=True, deep_stem=True):
    assert model_depth in [18, 34, 50, 101 ,152]
    if model_depth == 50:
        model = ResNeSt(BottleneckBlock, [3, 4, 6, 3], init_width, block_width, n_class, in_ch, r, init_down, deep_stem)
    elif model_depth == 101:
        model = ResNeSt(BottleneckBlock, [3, 4, 23, 3], init_width, block_width, n_class, in_ch, r, init_down, deep_stem)
    elif model_depth == 152:
        model = ResNeSt(BottleneckBlock, [3, 8, 36, 3], init_width, block_width, n_class, in_ch, r, init_down, deep_stem)
    return model
        
        