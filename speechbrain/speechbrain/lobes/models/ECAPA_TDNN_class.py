"""A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020
"""

# import os
import torch  # noqa: F401
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear


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


class ECAPA_TDNN(torch.nn.Module):
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
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

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

    # def forward(self, x, lengths=None):
    #     """Returns the embedding vector.
    #
    #     Arguments
    #     ---------
    #     x : torch.Tensor
    #         Tensor of shape (batch, time, channel).
    #     """
    #     # Minimize transpose for efficiency
    #     x = x.transpose(1, 2)
    #
    #     xl = []
    #     for layer in self.blocks:
    #         try:
    #             x = layer(x, lengths=lengths)
    #         except TypeError:
    #             x = layer(x)
    #         xl.append(x)
    #
    #     # Multi-layer feature aggregation
    #     x = torch.cat(xl[1:], dim=1)
    #     x = self.mfa(x)
    #
    #     # Attentive Statistical Pooling
    #     x = self.asp(x, lengths=lengths)
    #     x = self.asp_bn(x)
    #
    #     # Final linear transformation
    #     x = self.fc(x)
    #     y= F.normalize(self.head(x), dim=1)
    #     x = x.transpose(1, 2)
    #     return x,y
    def forward(self, x,lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
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
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

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

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
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
        y= F.normalize(self.head(x), dim=1)
        x = x.transpose(1, 2)
        return x,y
class class_generator(torch.nn.Module):
    def __init__(self,
        weight_generator_type,
        classifier_type,
        nclass,
        nFeat,
        scale_cls,
        scale_att,
        ):
        super(class_generator, self).__init__()
        self.weight_generator_type = weight_generator_type  # 权重生成器的种类
        self.classifier_type = classifier_type  # 分类器的种类
        assert (self.classifier_type == 'cosine' or
                self.classifier_type == 'dotproduct')
        self.nclass = nclass
        self.nFeat = nFeat
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls),
            requires_grad=True)
        if self.weight_generator_type == 'none':
            # If the weight generator type is `none` then feature averaging is being used. However, in this case the generator does not
            # involve any learnable parameter and thus does not require training.
            self.favgblock = FeatExemplarAvgBlock(nFeat)  # 特征平均，此时不需要训练
        elif self.weight_generator_type == 'feature_averaging':
            self.favgblock = FeatExemplarAvgBlock(nFeat)
            self.wnLayerFavg = LinearDiag(nFeat)
        elif self.weight_generator_type == 'attention_based':
            # attention权重
            self.favgblock = FeatExemplarAvgBlock(nFeat)  # 先求一个平均特征
            self.attblock = AttentionBasedBlock(
                nFeat, nclass, scale_att=scale_att)  # 再求一个attention特征
            self.wnLayerFavg = LinearDiag(nFeat)
            self.wnLayerWatt = LinearDiag(nFeat)
        else:
            raise ValueError('Not supported/recognized type {0}'.format(
                self.weight_generator_type))
    def get_classification_weights(
            self, Kbase_ids, features_train=None, labels_train=None):
        """Gets the classification weights of the base and novel categories.

        This routine returns the classification weight of the base categories
        and also (if training data, i.e., features_train and labels_train, for
        the novel categories are provided) of the novel categories.

        Args:
            Kbase_ids: A 2D tensor with shape [batch_size x nKbase] that for
                each training episode in the the batch it includes the indices
                of the base categories that are being used. `batch_size` is the
                number of training episodes in the batch and `nKbase` is the
                number of base categories.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the features of the training examples of each training episode
                in the batch. `num_train_examples` is the number of train
                examples in each training episode. Those training examples are
                from the novel categories.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the labels (encoded as 1-hot vectors of lenght nKnovel) of the
                training examples of each training episode in the batch.
                `nKnovel` is the number of novel categories.

        Returns:
            cls_weights: A 3D tensor of shape [batch_size x nK x num_channels]
                that includes the classification weight vectors
                (of `num_channels` length) of categories involved on each
                training episode in the batch. If training data for the novel
                categories are provided (i.e., features_train or labels_train
                are None) then cls_weights includes only the classification
                weights of the base categories; in this case nK is equal to
                nKbase. Otherwise, cls_weights includes the classification
                weights of both base and novel categories; in this case nK is
                equal to nKbase + nKnovel.
        """

        #***********************************************************************
        #******** Get the classification weights for the base categories *******
        batch_size, nKbase = Kbase_ids.size()#[batch_size x nKbase]
        weight_base = self.weight_base[Kbase_ids.view(-1)]
        #输出
        # Kbase_ids.view(-1)[(batch_size x nKbase)*1]
        #view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        weight_base = weight_base.view(batch_size, nKbase, -1)

        #***********************************************************************

        if features_train is None or labels_train is None:
            # If training data for the novel categories are not provided then
            # return only the classification weights of the base categories.
            return weight_base

        #***********************************************************************
        #******* Generate classification weights for the novel categories ******
        _, num_train_examples, num_channels = features_train.size()#[batch_size x num_train_examples x num_channels]
        nKnovel = labels_train.size(2)
        if self.classifier_type=='cosine':
            features_train = F.normalize(
                features_train, p=2, dim=features_train.dim()-1, eps=1e-12)
        if self.weight_generator_type=='none':
            weight_novel = self.favgblock(features_train, labels_train)
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='feature_averaging':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        elif self.weight_generator_type=='attention_based':
            weight_novel_avg = self.favgblock(features_train, labels_train)
            weight_novel_avg = self.wnLayerFavg(
                weight_novel_avg.view(batch_size * nKnovel, num_channels)
            )
            if self.classifier_type=='cosine':
                weight_base_tmp = F.normalize(
                    weight_base, p=2, dim=weight_base.dim()-1, eps=1e-12)
            else:
                weight_base_tmp = weight_base

            weight_novel_att = self.attblock(
                features_train, labels_train, weight_base_tmp, Kbase_ids)
            weight_novel_att = self.wnLayerWatt(
                weight_novel_att.view(batch_size * nKnovel, num_channels)
            )
            weight_novel = weight_novel_avg + weight_novel_att
            weight_novel = weight_novel.view(batch_size, nKnovel, num_channels)
        else:
            raise ValueError('Not supported / recognized type {0}'.format(
                self.weight_generator_type))
        #***********************************************************************

        # Concatenate the base and novel classification weights and return them.
        weight_both = torch.cat([weight_base, weight_novel], dim=1)
        # weight_both shape: [batch_size x (nKbase + nKnovel) x num_channels]

        return weight_both


    def apply_classification_weights(self, features, cls_weights):
        """Applies the classification weight vectors to the feature vectors.

        Args:
            features: A 3D tensor of shape
                [batch_size x num_test_examples x num_channels] with the feature
                vectors (of `num_channels` length) of each example on each
                trainining episode in the batch. `batch_size` is the number of
                training episodes in the batch and `num_test_examples` is the
                number of test examples of each training episode.
            cls_weights: A 3D tensor of shape [batch_size x nK x num_channels]
                that includes the classification weight vectors
                (of `num_channels` length) of the `nK` categories used on
                each training episode in the batch. `nK` is the number of
                categories (e.g., the number of base categories plus the number
                of novel categories) used on each training episode.

        Return:
            cls_scores: A 3D tensor with shape
                [batch_size x num_test_examples x nK] that represents the
                classification scores of the test examples for the `nK`
                categories.
        """
        if self.classifier_type=='cosine':
            features = F.normalize(
                features, p=2, dim=features.dim()-1, eps=1e-12)
            cls_weights = F.normalize(
                cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0,
            self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
        return cls_scores


    def forward(self, features_test, Kbase_ids, features_train=None, labels_train=None):
        #8*30    8*59
        """Recognize on the test examples both base and novel categories.

        Recognize on the test examples (i.e., `features_test`) both base and
        novel categories using the approach proposed on our CVPR2018 paper
        "Dynamic Few-Shot Visual Learning without Forgetting". In order to
        classify the test examples the provided training data for the novel
        categories (i.e., `features_train` and `labels_train`) are used in order
        to generate classification weight vectors of those novel categories and
        then those classification weight vectors are applied on the features of
        the test examples.

        Args:
            features_test: A 3D tensor with shape
                [batch_size x num_test_examples x num_channels] that represents
                the features of the test examples each training episode in the
                batch. Those examples can come both from base and novel
                categories. `batch_size` is the number of training episodes in
                the batch, `num_test_examples` is the number of test examples
                in each training episode, and `num_channels` is the number of
                feature channels.#batch是speaker数量，num_test_exm每个spk的样本，nchannel数量
            Kbase_ids: A 2D tensor with shape [batch_size x nKbase] that for
                each training episode in the the batch it includes the indices
                of the base categories that are being used. `nKbase` is the
                number of base categories.nkbase的基本类别
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the features of the training examples of each training episode
                 in the batch. `num_train_examples` is the number of train
                examples in each training episode. Those training examples are
                from the novel categories. If features_train is None then the
                current function will only return the classification scores for
                the base categories.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the labels (encoded as 1-hot vectors of lenght nKnovel) of the
                training examples of each training episode in the batch.
                `nKnovel` is the number of novel categories. If labels_train is
                None then the current function will return only the
                classification scores for the base categories.

        Return:
            cls_scores: A 3D tensor with shape
                [batch_size x num_test_examples x (nKbase + nKnovel)] that
                represents the classification scores of the test examples
                for the nKbase and nKnovel novel categories. If features_train
                or labels_train are None the only the classification scores of
                the base categories are returned. In that case the shape of
                cls_scores is [batch_size x num_test_examples x nKbase].
        """
        cls_weights = self.get_classification_weights(
            Kbase_ids, features_train, labels_train)
        cls_scores = self.apply_classification_weights(
            features_test, cls_weights)
        return cls_scores
class LinearDiag(torch.nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)#新生成的权重

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)#对应的偏置 fill将tensor中所有值都填充为指定值
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)
            #register_parameter(name, param)向我们建立的网络module添加 parameter

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))#X是两维的并且第1维的数量==权重的第0维（样本）
        out = X * self.weight.expand_as(X)#将特征乘以对应的权重（expand_as(X)将大小扩充维和X一样大的）
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out
class AttentionBasedBlock(torch.nn.Module):
    def __init__(self, nFeat, nK, scale_att=10.0):
        super(AttentionBasedBlock, self).__init__()
        self.nFeat = nFeat
        self.queryLayer = nn.Linear(nFeat, nFeat)
        self.queryLayer.weight.data.copy_(
            torch.eye(nFeat, nFeat) + torch.randn(nFeat, nFeat)*0.001)
        self.queryLayer.bias.data.zero_()

        self.scale_att = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_att), requires_grad=True)
        wkeys = torch.FloatTensor(nK, nFeat).normal_(0.0, np.sqrt(2.0/nFeat))
        self.wkeys = nn.Parameter(wkeys, requires_grad=True)


    def forward(self, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1) # [batch_size x nKbase x num_features]
        labels_train_transposed = labels_train.transpose(1,2)
        nKnovel = labels_train_transposed.size(1) # [batch_size x nKnovel x num_train_examples]

        features_train = features_train.view(
            batch_size*num_train_examples, num_features)
        Qe = self.queryLayer(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.nFeat)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim()-1, eps=1e-12)

        wkeys = self.wkeys[Kbase.view(-1)] # the keys of the base categoreis
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim()-1, eps=1e-12)
        #Transpose from [batch_size x nKbase x nFeat] to
        #[batch_size x self.nFeat x nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.nFeat).transpose(1,2)

        # Compute the attention coeficients
        # batch matrix multiplications: AttentionCoeficients = Qe * wkeys ==>
        # [batch_size x num_train_examples x nKbase] =
        #   [batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoeficients = self.scale_att * torch.bmm(Qe, wkeys)
        s=nn.Softmax(dim=1)
        AttentionCoeficients = F.softmax(
            AttentionCoeficients.view(batch_size*num_train_examples, nKbase))
        AttentionCoeficients = AttentionCoeficients.view(
            batch_size, num_train_examples, nKbase)

        # batch matrix multiplications: weight_novel = AttentionCoeficients * weight_base ==>
        # [batch_size x num_train_examples x num_features] =
        # [batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoeficients, weight_base)
        # batch matrix multiplications: weight_novel = labels_train_transposed * weight_novel ==>
        # [batch_size x nKnovel x num_features] =
        # [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel
class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1,2)#调转1,2维
        weight_novel = torch.bmm(labels_train_transposed, features_train)#bmm第一维相同，计算后面两维的矩阵乘法
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))#张量和标量做逐元素除法
        return weight_novel
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
        weight = torch.FloatTensor(out_neurons, input_size, device=device)
        self.weight_base = nn.Parameter(weight)
        nn.init.xavier_uniform_(self.weight_base)


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
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight_base))
        return x.unsqueeze(1)
