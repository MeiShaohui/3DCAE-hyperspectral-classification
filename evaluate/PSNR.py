from __future__ import division, print_function, absolute_import

from sklearn.metrics import mean_squared_error

import numpy as np
import math


def psnr(x_true, x_pred):
    n_samples = x_true.shape[0]
    n_bands = x_true.shape[1]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    for k in range(n_bands):
        x_true_k = x_true[:, k].reshape([-1])
        x_pred_k = x_pred[:, k].reshape([-1])
        MSE[k] = 1.0 / n_samples * mean_squared_error(x_true_k, x_pred_k, )
        MAX_k = np.max(x_true_k)
        if MAX_k != 0:
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
        else:
            mask[k] = 0

    psnr = PSNR.sum()/mask.sum()
    mse = MSE.mean()
    print('psnr', psnr)
    print('mse', mse)
    return psnr, mse
