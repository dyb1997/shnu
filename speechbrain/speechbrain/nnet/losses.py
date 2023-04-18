"""
Losses for training neural networks.

Authors
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Hwidong Na 2020
 * Yan Gao 2020
 * Titouan Parcollet 2020
"""
import math
import torch
import logging
import functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from speechbrain.utils.Accuracy import accuracy
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.decoders.ctc import filter_ctc_output
import time, pdb, numpy
#from utils import accuracy

logger = logging.getLogger(__name__)


# class LossFunction(nn.Module):
#     def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
#         super(LossFunction, self).__init__()
#
#         self.test_normalize = True
#
#         self.w = nn.Parameter(torch.tensor(init_w))
#         self.b = nn.Parameter(torch.tensor(init_b))
#         self.criterion = torch.nn.CrossEntropyLoss()
#
#         print('Initialised AngleProto')
#
#     def forward(self, x, label=None):
#         #x[batch*1*192]
#         # assert x.size()[1] >= 2
#         print(x.shape)
#         #print(y.shape)
#         out_anchor = x[:,0,:]
#         out_positive = x[:,1,:]
#         stepsize = out_anchor.size()[0]
#         cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
#         torch.clamp(self.w, 1e-6)
#         cos_sim_matrix = cos_sim_matrix * self.w + self.b
#
#         label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
#         #数组转换成张量，
#         nloss = self.criterion(cos_sim_matrix, label)
#         # prec1 = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]
#
#         return nloss,label
class LossFunction(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        out_anchor = torch.mean(x[:, 1:, :], 1)  # 第一个anchor的
        out_positive = x[:, 0, :]  # positive对
        stepsize = out_anchor.size()[0]
        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        #数组转换成张量，
        nloss = self.criterion(cos_sim_matrix, label)
        # prec1 = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss,label
def transducer_loss(
    log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"
):
    """Transducer loss, see `speechbrain/nnet/loss/transducer_loss.py`.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, maxT, maxU, num_labels].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len].
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the label indices.
    reduction : str
        Specifies the reduction to apply to the output: 'mean' | 'batchmean' | 'sum'.
    """
    from speechbrain.nnet.loss.transducer_loss import Transducer

    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    return Transducer.apply(
        log_probs, targets, input_lens, target_lens, blank_index, reduction
    )


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        Torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)

        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        """
        Arguments
        ---------
        tensor : torch.Tensor
            Tensor to reorder given the optimal permutation, of shape
            [batch, ..., sources].
        p : list of tuples
            List of optimal permutations, e.g. for batch=2 and n_sources=3
            [(0, 1, 2), (0, 2, 1].

        Returns
        -------
        reordered : torch.Tensor
            Reordered tensor given permutation p.
        """

        reordered = torch.zeros_like(tensor, device=tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss : torch.Tensor
                Permutation invariant loss for current examples, tensor of
                shape [batch]

            perms : list
                List of indexes for optimal permutation of the inputs over
                sources.
                e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def ctc_loss(
    log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"
):
    """CTC loss.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'batch',
        'batchmean', 'none'.
        See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batchmean":
        reduction_loss = "sum"
    elif reduction == "batch":
        reduction_loss = "none"
    else:
        reduction_loss = reduction
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
        reduction=reduction_loss,
    )

    if reduction == "batchmean":
        return loss / targets.shape[0]
    elif reduction == "batch":
        N = loss.size(0)
        return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
    else:
        return loss


def l1_loss(
    predictions, targets, length=None, allowed_len_diff=3, reduction="mean"
):
    """Compute the true l1 loss, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor with the same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> l1_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.1000)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.l1_loss, reduction="none")
    return compute_masked_loss(
        loss, predictions, targets, length, reduction=reduction
    )


def mse_loss(
    predictions, targets, length=None, allowed_len_diff=3, reduction="mean"
):
    """Compute the true mean squared error, accounting for length differences.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape ``[batch, time, *]``.
    targets : torch.Tensor
        Target tensor with the same size as predicted tensor.
    length : torch.Tensor
        Length of each utterance for computing true error with a mask.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1, 0.1, 0.9]])
    >>> mse_loss(probs, torch.tensor([[1., 0., 0., 1.]]))
    tensor(0.0100)
    """
    predictions, targets = truncate(predictions, targets, allowed_len_diff)
    loss = functools.partial(torch.nn.functional.mse_loss, reduction="none")
    return compute_masked_loss(
        loss, predictions, targets, length, reduction=reduction
    )


def classification_error(
    probabilities, targets, length=None, allowed_len_diff=3, reduction="mean"
):
    """Computes the classification error at frame or batch level.

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob]
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames]
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> classification_error(probs, torch.tensor([1, 1]))
    tensor(0.5000)
    """
    if len(probabilities.shape) == 3 and len(targets.shape) == 2:
        probabilities, targets = truncate(
            probabilities, targets, allowed_len_diff
        )

    def error(predictions, targets):
        predictions = torch.argmax(probabilities, dim=-1)
        return (predictions != targets).float()

    return compute_masked_loss(
        error, probabilities, targets.long(), length, reduction=reduction
    )


def nll_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    reduction="mean",
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> nll_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if len(log_probabilities.shape) == 3:
        log_probabilities, targets = truncate(
            log_probabilities, targets, allowed_len_diff
        )
        log_probabilities = log_probabilities.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(torch.nn.functional.nll_loss, reduction="none")
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def bce_loss(
    inputs,
    targets,
    length=None,
    weight=None,
    pos_weight=None,
    reduction="mean",
    allowed_len_diff=3,
    label_smoothing=0.0,
):
    """Computes binary cross-entropy (BCE) loss. It also applies the sigmoid
    function directly (this improves the numerical stability).

    Arguments
    ---------
    inputs : torch.Tensor
        The output before applying the final softmax
        Format is [batch[, 1]?] or [batch, frames[, 1]?].
        (Works with or without a singleton dimension at the end).
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    weight : torch.Tensor
        A manual rescaling weight if provided it’s repeated to match input
        tensor shape.
    pos_weight : torch.Tensor
        A weight of positive examples. Must be a vector with length equal to
        the number of classes.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction: str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> inputs = torch.tensor([10.0, -6.0])
    >>> targets = torch.tensor([1, 0])
    >>> bce_loss(inputs, targets)
    tensor(0.0013)
    """
    # Squeeze singleton dimension so inputs + targets match
    if len(inputs.shape) == len(targets.shape) + 1:
        inputs = inputs.squeeze(-1)

    # Make sure tensor lengths match
    if len(inputs.shape) >= 2:
        inputs, targets = truncate(inputs, targets, allowed_len_diff)
    elif length is not None:
        raise ValueError("length can be passed only for >= 2D inputs.")

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(
        torch.nn.functional.binary_cross_entropy_with_logits,
        weight=weight,
        pos_weight=pos_weight,
        reduction="none",
    )
    return compute_masked_loss(
        loss,
        inputs,
        targets.float(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def kldiv_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    pad_idx=0,
    reduction="mean",
):
    """Computes the KL-divergence error at the batch level.
    This loss applies label smoothing directly to the targets

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> kldiv_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if label_smoothing > 0:
        if log_probabilities.dim() == 2:
            log_probabilities = log_probabilities.unsqueeze(1)

        bz, time, n_class = log_probabilities.shape
        targets = targets.long().detach()

        confidence = 1 - label_smoothing

        log_probabilities = log_probabilities.view(-1, n_class)
        targets = targets.view(-1)
        with torch.no_grad():
            true_distribution = log_probabilities.clone()
            true_distribution.fill_(label_smoothing / (n_class - 1))
            ignore = targets == pad_idx
            targets = targets.masked_fill(ignore, 0)
            true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

        loss = torch.nn.functional.kl_div(
            log_probabilities, true_distribution, reduction="none"
        )
        loss = loss.masked_fill(ignore.unsqueeze(1), 0)

        # return loss according to reduction specified
        if reduction == "mean":
            return loss.sum().mean()
        elif reduction == "batchmean":
            return loss.sum() / bz
        elif reduction == "batch":
            return loss.view(bz, -1).sum(1) / length
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        return nll_loss(log_probabilities, targets, length, reduction=reduction)


def truncate(predictions, targets, allowed_len_diff=3):
    """Ensure that predictions and targets are the same length.

    Arguments
    ---------
    predictions : torch.Tensor
        First tensor for checking length.
    targets : torch.Tensor
        Second tensor for checking length.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    """
    len_diff = predictions.shape[1] - targets.shape[1]
    if len_diff == 0:
        return predictions, targets
    elif abs(len_diff) > allowed_len_diff:
        raise ValueError(
            "Predictions and targets should be same length, but got %s and "
            "%s respectively." % (predictions.shape[1], targets.shape[1])
        )
    elif len_diff < 0:
        return predictions, targets[:, : predictions.shape[1]]
    else:
        return predictions[:, : targets.shape[1]], targets


def compute_masked_loss(
    loss_fn,
    predictions,
    targets,
    length=None,
    label_smoothing=0.0,
    reduction="mean",
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none").
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output
        Distributions. https://arxiv.org/abs/1701.06548
    reduction : str
        One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
        single value and 'batch' returns one per item in the batch and
        'batchmean' is sum / batch_size and 'none' returns all.
    """
    mask = torch.ones_like(targets)
    if length is not None:
        length_mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )

        # Handle any dimensionality of input
        while len(length_mask.shape) < len(mask.shape):
            length_mask = length_mask.unsqueeze(-1)
        length_mask = length_mask.type(mask.dtype)
        mask *= length_mask

    # Compute, then reduce loss
    loss = loss_fn(predictions, targets) * mask
    N = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / N
    elif reduction == "batch":
        loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

    if label_smoothing == 0:
        return loss
    else:
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss


def get_si_snr_with_pitwrapper(source, estimate_source):
    """This function wraps si_snr calculation with the speechbrain pit-wrapper.

    Arguments:
    ---------
    source: [B, T, C],
        Where B is the batch size, T is the length of the sources, C is
        the number of sources the ordering is made so that this loss is
        compatible with the class PitWrapper.

    estimate_source: [B, T, C]
        The estimated source.

    Example:
    ---------
    >>> x = torch.arange(600).reshape(3, 100, 2)
    >>> xhat = x[:, :, (1, 0)]
    >>> si_snr = -get_si_snr_with_pitwrapper(x, xhat)
    >>> print(si_snr)
    tensor([135.2284, 135.2284, 135.2284])
    """

    pit_si_snr = PitWrapper(cal_si_snr)
    loss, perms = pit_si_snr(source, estimate_source)

    return loss


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity
    scale : float
        The scale for cosine similiarity

    Return
    ---------
    predictions : torch.Tensor

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity.
    scale: float
        The scale for cosine similiarity.

    Returns
    -------
    predictions : torch.Tensor
        Tensor.
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    Returns
    ---------
    loss : torch.Tensor
        Learning loss
    predictions : torch.Tensor
        Log probabilities
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


def ctc_loss_kd(log_probs, targets, input_lens, blank_index, device):
    """Knowledge distillation for CTC loss.

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    log_probs : torch.Tensor
        Predicted tensor from student model, of shape [batch, time, chars].
    targets : torch.Tensor
        Predicted tensor from single teacher model, of shape [batch, time, chars].
    input_lens : torch.Tensor
        Length of each utterance.
    blank_index : int
        The location of the blank symbol among the character indexes.
    device : str
        Device for computing.
    """
    scores, predictions = torch.max(targets, dim=-1)

    pred_list = []
    pred_len_list = []
    for j in range(predictions.shape[0]):
        # Getting current predictions
        current_pred = predictions[j]

        actual_size = (input_lens[j] * log_probs.shape[1]).round().int()
        current_pred = current_pred[0:actual_size]
        current_pred = filter_ctc_output(
            list(current_pred.cpu().numpy()), blank_id=blank_index
        )
        current_pred_len = len(current_pred)
        pred_list.append(current_pred)
        pred_len_list.append(current_pred_len)

    max_pred_len = max(pred_len_list)
    for j in range(predictions.shape[0]):
        diff = max_pred_len - pred_len_list[j]
        for n in range(diff):
            pred_list[j].append(0)

    # generate soft label of teacher model
    fake_lab = torch.from_numpy(np.array(pred_list))
    fake_lab.to(device)
    fake_lab = fake_lab.int()
    fake_lab_lengths = torch.from_numpy(np.array(pred_len_list)).int()
    fake_lab_lengths.to(device)

    input_lens = (input_lens * log_probs.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        fake_lab,
        input_lens,
        fake_lab_lengths,
        blank_index,
        zero_infinity=True,
    )


def ce_kd(inp, target):
    """Simple version of distillation for cross-entropy loss.

    Arguments
    ---------
    inp : torch.Tensor
        The probabilities from student model, of shape [batch_size * length, feature]
    target : torch.Tensor
        The probabilities from teacher model, of shape [batch_size * length, feature]
    """
    return (-target * inp).sum(1)


def nll_loss_kd(
    probabilities, targets, rel_lab_lengths,
):
    """Knowledge distillation for negative log-likelihood loss.

    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310

    Arguments
    ---------
    probabilities : torch.Tensor
        The predicted probabilities from the student model.
        Format is [batch, frames, p]
    targets : torch.Tensor
        The target probabilities from the teacher model.
        Format is [batch, frames, p]
    rel_lab_lengths : torch.Tensor
        Length of each utterance, if the frame-level loss is desired.

    Example
    -------
    >>> probabilities = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    >>> targets = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    >>> rel_lab_lengths = torch.tensor([1.])
    >>> nll_loss_kd(probabilities, targets, rel_lab_lengths)
    tensor(-0.7400)
    """
    # Getting the number of sentences in the minibatch
    N_snt = probabilities.shape[0]

    # Getting the maximum length of label sequence
    max_len = probabilities.shape[1]

    # Getting the label lengths
    lab_lengths = torch.round(rel_lab_lengths * targets.shape[1]).int()

    # Reshape to [batch_size * length, feature]
    prob_curr = probabilities.reshape(N_snt * max_len, probabilities.shape[-1])

    # Generating mask
    mask = length_to_mask(
        lab_lengths, max_len=max_len, dtype=torch.float, device=prob_curr.device
    )

    # Reshape to [batch_size * length, feature]
    lab_curr = targets.reshape(N_snt * max_len, targets.shape[-1])

    loss = ce_kd(prob_curr, lab_curr)
    # Loss averaging
    loss = torch.sum(loss.reshape(N_snt, max_len) * mask) / torch.sum(mask)
    return loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        print(features.shape)
        # feature（bc ， channel，1）16,2,192
        batch_size = features.shape[0]
        #（16，1）
        # print(batch_size)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            #torch.eye这个函数主要是为了生成对角线全1，其余部分全0的二维数组
            #16，1
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # print(labels)
            #tensor([[611],
                    # [450],
                    # [ 24],
                    # [329],
                    # [294],
                    # [196],
                    # [377],
                    # [  2],
                    # [251],
                    # [ 87],
                    # [185],
                    # [164],
                    # [140],
                    # [110],
                    # [113],
                    # [201]], device='cuda:0')
            #当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
            # a = torch.arange(0, 20)  # 此时a的shape是(1,20)
            # a.view(4, 5).shape  # 输出为(4,5)
            # a.view(-1, 5).shape  # 输出为(4,5)
            # a.view(4, -1).shape  # 输出为(4,5)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(mask)
            # tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]],
            #        device='cuda:0')

            # torch.eq对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(features.shape[1])
        # contrast_count=2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape)
        # [32,192]
        # tensor([[2.1991, 0.3406, 0.6839, ..., -1.7272, 1.5392, -0.4981],
        #         [0.6755, -0.2010, 0.4706, ..., -0.8345, 0.6629, 0.4647],
        #         [-1.4394, -0.0903, -0.6265, ..., 0.8543, -0.8075, -0.0281],
        #         ...,
        #         [-0.9926, -0.7552, -0.1552, ..., 0.9917, -0.7796, 0.3357],
        #         [1.3102, 0.6308, 0.4973, ..., -1.7667, 0.2753, 0.4175],
        #         [-0.2377, 0.2948, -0.1453, ..., 0.3032, 0.2850, -0.3460]],
        #        device='cuda:0', grad_fn= < CatBackward >)
        # torch.unbind()
        # 移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片。torch.unbind(input, dim=0) → seq
        # torch.cat（）按dim 拼接
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            # [32,192]
            anchor_count = contrast_count
            # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print(torch.matmul(anchor_feature, contrast_feature.T).shape)[32,32]
        # tensor([[227.2883, 30.0309, -90.6780, ..., -89.7573, 46.6995, -17.7273],
        #         [30.0309, 76.5992, -24.4581, ..., -12.6905, 11.0980, -5.7315],
        #         [-90.6780, -24.4581, 67.4853, ..., 40.7129, -34.3807, -2.0669],
        #         ...,
        #         [-89.7573, -12.6905, 40.7129, ..., 57.7119, -22.3954, 5.0620],
        #         [46.6995, 11.0980, -34.3807, ..., -22.3954, 73.1069, -7.5220],
        #         [-17.7273, -5.7315, -2.0669, ..., 5.0620, -7.5220, 42.5240]],
        #        device='cuda:0', grad_fn= < MmBackward >)

        # torch.div(input, other)将input张量中值除以other张量中对应的元素值，other可为张量也可为标量
        # torch.matmul是tensor的乘法，输入可以是高维的。当输入是都是二维时，就是普通的矩阵乘法
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # 选出每个样本对应的最大样本值logits最大样本值，_索引(对角线)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits)原值减去对角线的值

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # [16,16]->[32,32]
        #
        # mask-out self-contrast cases
        # torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False)
        # → Tensor返回一个填充了标量值1的张量，其大小与之相同input
        # torch,arrage(a,b)a间隔，b截止
        # print(torch.arange(batch_size * anchor_count).view(-1, 1).to(device))
        # 去掉对角线的值
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # scatter(dim, index, src)的参数有3个
        # dim：沿着哪个维度进行索引
        # index：用来scatter的元素索引
        # src：用来scatter的源元素，可以是一个标量或一个张量
        mask = mask * logits_mask
        # print(mask)

        # compute log_prob
        print()
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        print(mask.sum(1))
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss