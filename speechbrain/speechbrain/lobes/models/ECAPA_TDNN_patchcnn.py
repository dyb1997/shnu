"""A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020
"""

import os,sys
import numpy as np
from enum import Enum
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.CNN import Conv2d as _Conv2d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.normalization import BatchNorm2d as _BatchNorm2d
from speechbrain.nnet.linear import Linear


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)
class Conv2d(_Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class BatchNorm2d(_BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

class BatchNorm1d(_BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class D2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        stride,
        activation=nn.ReLU,
        groups=1,
    ):
        super(D2Block, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm2d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))

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

class TDNNBlockpatch(nn.Module):
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
        super(TDNNBlockpatch, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.patchup = PatchUp()

    def forward(self, x,spkid,stage):

        r = np.random.rand(1)
        if r < 0.7 and stage == sb.Stage.TRAIN:
            arget_a, target_b, target_reweighted, x, total_unchanged_portion = self.patchup(self.conv(x), spkid, lam=None)
        else:
            x = x
            arget_a = spkid
            target_b, target_reweighted, total_unchanged_portion = None, None, None
        return self.norm(self.activation(x)),arget_a,target_b,target_reweighted,total_unchanged_portion
class Res2NetBlock2D(torch.nn.Module):
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
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock2D, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                D2Block(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=(1,1)
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        # print(x.shape,"890")#[8, 40, 201, 128]
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=3)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
                # print(y_i.shape,"121111")
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=3)
        return y
class Res2NetBlock(torch.nn.Module):
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
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x

class SEBlock2D(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.line1=nn.Linear(40, 40 // 8)
        self.relu = torch.nn.ReLU(inplace=True)
        self.line2=nn.Linear(40 // 8,40)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        b,c,f,t = x.size()#btfc
        #[8, 40, 201, 128]
        out = self.avg_pool(x).view(b, c)
        # print(out.shape,"se")
        s = self.relu(self.line1(out))
        s = self.sigmoid(self.line2(s)).view(b, c, 1, 1)

        return s.expand_as(x) * x


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

class SERes2NetBlock2d(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    [B*40*T*c]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        stride=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = D2Block(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            stride=stride,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock2D(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = D2Block(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            stride=stride,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock2D(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)
        x = self.tdnn1(x)
        # print(x.shape,"tdnn1")#8,40,201,128
        x = self.res2net_block(x)
        # print(x.shape,"110")
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual
class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual

class ECAPA_TDNNcnn(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[128, 128, 128, 128, 512, 512, 512, 512, 1536],
        kernel_sizes=[ 3, 3, 3, 3, 5, 3, 3, 3, 1],
        dilations=[1, 1, 1, 1, 1, 2, 3, 4, 1],
        stride = [2,1,1,2],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1,1,1,1,1, 1, 1, 1, 1],
    ):

        super().__init__()
        print(kernel_sizes)
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks2D = nn.ModuleList()
        self.blocks = nn.ModuleList()
        #第一层2D卷积（B*80*T*1）[8, 1, 201,80]
        # self.blocks2D.append(
        self.b1=D2Block(
                1,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                (1,stride[0]),
                activation,
                groups[0],
            )
        # )
        #加入两层2D-se-res2block（B*40*T*c）
        for i in range(1,3):
            self.blocks2D.append(
                SERes2NetBlock2d(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    stride=stride[i],
                    activation=activation,
                    groups=groups[i],
                )
            )
        # 再来一层2Dcnn（B*40*T*C）
        # self.blocks2D.append(
        self.b2=D2Block(
                channels[3],
                channels[3],
                kernel_sizes[3],
                dilations[3],
                (1,stride[3]),
                activation,
                groups[3],
            )
        # )
        #使用flatten channel和frequency变成（B*(C*20)*T）
        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[4],
                kernel_sizes[4],
                dilations[4],
                activation,
                groups[4],
            )
        )

        # SE-Res2Net layers
        for i in range(5, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x,lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)#(B*T*F*1)[8, 80, 201, 1]
        x=x.unsqueeze(-1)
        x=self.b1(x)
        print(x.shape)
        # expectedinput[8, 1, 203, 82]
        for layer in self.blocks2D:
            try:
                x=layer(x,lengths)
            except TypeError:
                x=layer(x)
#8, 40, 201, 128
        x=self.b2(x)
        x=x.permute(0,2,3,1)
        x=x.flatten(2)
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x
class ECAPA_TDNNcnnpatch(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[128, 128, 128, 128, 512, 512, 512, 512, 1536],
        kernel_sizes=[ 3, 3, 3, 3, 5, 3, 3, 3, 1],
        dilations=[1, 1, 1, 1, 1, 2, 3, 4, 1],
        stride = [2,1,1,2],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1,1,1,1,1, 1, 1, 1, 1],
    ):

        super().__init__()
        print(kernel_sizes)
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks2D = nn.ModuleList()
        self.blocks = nn.ModuleList()
        #第一层2D卷积（B*80*T*1）[8, 1, 201,80]
        # self.blocks2D.append(
        self.b1=D2Block(
                1,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                (1,stride[0]),
                activation,
                groups[0],
            )
        # )
        #加入两层2D-se-res2block（B*40*T*c）
        for i in range(1,3):
            self.blocks2D.append(
                SERes2NetBlock2d(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    stride=stride[i],
                    activation=activation,
                    groups=groups[i],
                )
            )
        # 再来一层2Dcnn（B*40*T*C）
        # self.blocks2D.append(
        self.b2=D2Block(
                channels[3],
                channels[3],
                kernel_sizes[3],
                dilations[3],
                (1,stride[3]),
                activation,
                groups[3],
            )
        # )
        #使用flatten channel和frequency变成（B*(C*20)*T）
        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[4],
                kernel_sizes[4],
                dilations[4],
                activation,
                groups[4],
            )
        )

        # SE-Res2Net layers
        for i in range(5, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )
        self.patch = PatchUp()
    def forward(self, x,spkid,stage,lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)#(B*T*F*1)[8, 80, 201, 1]
        x=x.unsqueeze(-1)
        x=self.b1(x)
        # x, arget_a, target_b, target_reweighted, total_unchanged_portion = self.tdnnblockpatch(x, spkid, stage)
        r = np.random.rand(1)
        if r < 0.7 and stage == sb.Stage.TRAIN:
            arget_a, target_b, target_reweighted, x, total_unchanged_portion = self.patch(x, spkid,lam = None)
        else:
            x = x
            arget_a  = spkid
            target_b, target_reweighted, total_unchanged_portion = None, None, None
        # expectedinput[8, 1, 203, 82]
        for layer in self.blocks2D:
            try:
                x=layer(x,lengths)
            except TypeError:
                x=layer(x)
#8, 40, 201, 128
        x=self.b2(x)
        x=x.permute(0,2,3,1)
        x=x.flatten(2)
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x,arget_a,target_b,target_reweighted,total_unchanged_portion
class ECAPA_TDNN2(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        # self.blocks.append(
        self.tdnnblockpatchup = TDNNBlockpatch(
            input_size,
            channels[0],
            kernel_sizes[0],
            dilations[0],
            activation,
            groups[0],
        )
        # )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )
        self.patchup  = PatchUp()
    def forward(self, x,spkid,stage, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        x,arget_a,target_b,target_reweighted,total_unchanged_portion=self.tdnnblockpatchup(x,spkid,stage)

        # r = np.random.rand(1)
        # if r < 0.7 and stage == sb.Stage.TRAIN:
        #     arget_a, target_b, target_reweighted, x, total_unchanged_portion = self.patchup(x, spkid,lam = None)
        # else:
        #     x = x
        #     arget_a  = spkid
        #     target_b, target_reweighted, total_unchanged_portion = None, None, None
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x,arget_a,target_b,target_reweighted,total_unchanged_portion
class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)



class PatchUpMode(Enum):
    SOFT = 'soft'
    HARD = 'hard'


class PatchUp(nn.Module):
    """
    PatchUp Module.
    This module is responsible for applying either Soft PatchUp or Hard PatchUp after a Convolutional module
    or convolutional residual block.
    """
    def __init__(self, block_size=7, gamma=0.9, patchup_type=PatchUpMode.SOFT):
        """
        PatchUp constructor.
        Args:
            block_size: An odd integer number that defines the size of blocks in the Mask that defines
            the continuous feature should be altered.
            gamma: It is float number in [0, 1]. The gamma in PatchUp decides the probability of altering a feature.
            patchup_type: It is an enum type of PatchUpMode. It defines PatchUp type that can be either
            Soft PatchUp or Hard PatchUp.
        """
        super(PatchUp, self).__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None

    def adjust_gamma(self, x):
        """
        The gamma in PatchUp decides the probability of altering a feature.
        This function is responsible to adjust the probability based on the
        gamma value since we are altering a continues blocks in feature maps.
        Args:
            x: feature maps for a minibatch generated by a convolutional module or convolutional layer.
        Returns:
            the gamma which is a float number in [0, 1]
        """
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, targets=None, lam=None, patchup_type=PatchUpMode.SOFT):
        """
            Forward pass in for the PatchUp Module.
        Args:
            x: Feature maps for a mini-batch generated by a convolutional module or convolutional layer.
            targets: target of samples in the mini-batch.
            lam: In a case, you want to apply PatchUp for a fixed lambda instead of sampling from the Beta distribution.
            patchup_type: either Hard PatchUp or Soft PatchUp.
        Returns:
            the interpolated hidden representation using PatchUp.
            target_a: targets associated with the first samples in the randomly selected sample pairs in the mini-batch.
            target_b: targets associated with the second samples in the randomly selected sample pairs in the mini-batch.
            target_reweighted: target target re-weighted for interpolated samples after altering patches
            with either Hard PatchUp or Soft PatchUp.
            x: interpolated hidden representations.
            total_unchanged_portion: the portion of hidden representation that remained unchanged after applying PatchUp.
        """
        self.patchup_type = patchup_type
        if type(self.training) == type(None):
            Exception("model's mode is not set in to neither training nor testing mode")
        if not self.training:
            # if the model is at the inference time (evaluation or test), we are not applying patchUp.
            return x, targets

        if type(lam) == type(None):
            # if we are not using fixed lambda, we should sample from the Beta distribution with fixed alpha equal to 2.
            # lambda will be a float number in range [0, 1].
            lam = np.random.beta(2.0, 2.0)

        if self.gamma_adj is None:
            self.gamma_adj = self.adjust_gamma(x)
        p = torch.ones_like(x[0]) * self.gamma_adj
        # For each feature in the feature map, we will sample from Bernoulli(p). If the result of this sampling
        # for feature f_{ij} is 0, then Mask_{ij} = 1. If the result of this sampling for f_{ij} is 1,
        # then the entire square region in the mask with the center Mask_{ij} and the width and height of
        # the square of block_size is set to 0.
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)

        # after creating the binary Mask. we are creating the binary Mask created for first sample as a pattern
        # for all samples in the minibatch as the PatchUp binary Mask. to do so, we can just expnand the pattern
        # created for the first sample.
        m_i_j = m_i_j.expand(x.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))

        # following line provides the continues blocks that should be altered with PatchUp denoted as holes here.
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)

        # following line gives the binary mask that contains 1 for the features that should be remain unchanged and 1
        # for the features that lie in the continues blocks that selected for interpolation.
        mask = 1 - holes
        unchanged = mask * x
        if mask_shape == 1:
            total_feats = x.size(1)
        else:
            total_feats = x.size(1) * (x.size(2) ** 2)
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        # following line gives the indices of second ssamples in the pair permuted randomly.
        indices = np.random.permutation(x.size(0))
        target_shuffled_onehot = targets[indices]
        patches = None
        target_reweighted = None
        target_b = None
        if self.patchup_type == PatchUpMode.SOFT:
            # apply Soft PatchUp combining operation for the selected continues blocks.
            target_reweighted = total_unchanged_portion * targets +  lam * total_changed_portion * targets + \
                                target_shuffled_onehot * (1 - lam) * total_changed_portion
            patches = holes * x
            patches = patches * lam + patches[indices] * (1 - lam)
            target_b = lam * targets + (1 - lam) * target_shuffled_onehot
        elif self.patchup_type == PatchUpMode.HARD:
            # apply Hard PatchUp combining operation for the selected continues blocks.
            target_reweighted = total_unchanged_portion * targets + total_changed_portion * target_shuffled_onehot
            patches = holes * x
            patches = patches[indices]
            target_b = targets[indices]
        x = unchanged + patches
        target_a = targets
        return target_a, target_b, target_reweighted, x, total_unchanged_portion
