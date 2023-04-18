"""A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear
import torchaudio


class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels
        self.log_input = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(1, 1))

        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                            hop_length=160, window_fn=torch.hamming_window,
                                                            n_mels=n_mels)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion * 2
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfb(x) + 1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1).detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.mean(x, dim=2, keepdim=True)

        if self.encoder_type == "SAP":
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))

class Res2NetL(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        num_filters,
        layers,
        nOut=512,
        encoder_type='SAP',
    ):
        super(Res2NetL, self).__init__()
        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size = (7,7), stride=(2, 1), padding=3, bias=False)
        #先过第一个二维卷积
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        # 2维的batchnorm
        self.relu = nn.ReLU(inplace=True)
        #激活函数
        #三层res层
        self.layer1 = self._make_layer(num_filters[0], layers[0])
        self.layer2 = self._make_layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(num_filters[3], layers[3], stride=(1, 1))
        #得到MFCC
        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * SEBasicBlock.expansion, num_filters[3] * SEBasicBlock.expansion)
            self.attention = self.new_parameter(num_filters[3] * SEBasicBlock.expansion, 1)
            out_dim = num_filters[3] * SEBasicBlock.expansion
        elif self.encoder_type == "ASP":
            self.sap_linear = nn.Linear(num_filters[3] * SEBasicBlock.expansion, num_filters[3] * SEBasicBlock.expansion)
            self.attention = self.new_parameter(num_filters[3] * SEBasicBlock.expansion, 1)
            out_dim = num_filters[3] * SEBasicBlock.expansion * 2
        else:
            raise ValueError('Undefined encoder')
        self.fc = nn.Linear(out_dim, nOut)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _make_layer(self,planes = 32,blocks = 32,stride =(2,2)):
        layers = []
        self.layer_module = None
        if stride != 1 or self.inplanes != planes:
            dowmsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                              kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers.append(
            SEBasicBlock(self.inplanes,planes,stride,dowmsample,
                        ))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                SEBasicBlock(self.inplanes,planes)
            )
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.mean(x, dim=2, keepdim=True)

        if self.encoder_type == "SAP":
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)
        elif self.encoder_type == "ASP":
            x = x.permute(0, 3, 1, 2).squeeze(-1)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        project = MLP(512,512,512)
        project = project(x)
        reproject = regu_MLP(512,512,128)
        reproject = reproject(project)
        return x,project,reproject


# class _make_layer(nn.Module):
#     def __init__(self,planes = 16,blocks = 64,stride =1,inplanes = 16, dowmsample = None):
#         super(_make_layer, self).__init__()
#         layers = []
#         self.layer_module = None
#         self.inplanes = inplanes
#         if stride != 1 or self.inplanes != planes:
#             dowmsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes),
#             )
#             layers.append(
#                 SEBasicBlock(self.inplanes,
#                              planes,
#                              stride,
#                              dowmsample,
#                              )
#             )
#             self.inplanes = planes
#             for i in range(1, blocks):
#                 layers.append(
#                     SEBasicBlock(self.inplanes,
#                                 planes)
#                 )
#         self.layer_module = nn.ModuleList(layers)
#     def forward(self,x):
#         xl = []
#         for layer in self.layer_module:
#             x = layer(x)
#             print(x)
#             xl.append(x)
#
#         print(xl)
#         return xl


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(residual)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=512,device="cuda"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.net = self.net.to(device)
    def forward(self, x):
        return self.net(x)
class regu_MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=128,device="cuda"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )
        self.net = self.net.to(device)
    def forward(self, x):
        return self.net(x)

class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats
