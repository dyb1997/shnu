import numpy as np


def exp_rampup(current, rampup_length):
    """Exponential rampup inspired by https://arxiv.org/abs/1610.02242
        Args:
            current: float, current step of the rampup
            rampup_length: float: length of the rampup

    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)# np.clip是一个截取函数,用于截取数组中小于或者大于某值的部分,并使得被截取部分等于固定值
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
