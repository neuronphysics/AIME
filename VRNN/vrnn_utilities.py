import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
import math
import time
import logging
import torch.nn as nn
from collections import OrderedDict
###Required for debugging to detect nan values in layers
def nan_hook(self, inp, out):
    """
    Check for NaN inputs or outputs at each layer in the model
    Usage:
        # forward hook
        for submodule in model.modules():
            submodule.register_forward_hook(nan_hook)
    """

    outputs = isinstance(out, tuple) and out or [out]
    inputs = isinstance(inp, tuple) and inp or [inp]
    contains_nan = lambda x: torch.isnan(x.data).any() if isinstance(x, nn.utils.rnn.PackedSequence) else lambda x: torch.isnan(x).any()
    #contains_nan = lambda x: torch.isnan(x).any()
    layer = self.__class__.__name__

    for i, inp in enumerate(inputs):
        if inp is not None and contains_nan(inp):
            raise RuntimeError(f'Found NaN input at index: {i} in layer: {layer}')

    for i, out in enumerate(outputs):
        if out is not None and contains_nan(out):
            raise RuntimeError(f'Found NaN output at index: {i} in layer: {layer}')

# import generic libraries
# computes the VAF (variance accounted for)
def compute_vaf(y, yhat, doprint=False):
    # reshape to ydim x -1
    num_outputs = y.shape[1]
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    diff = y - yhat
    num = np.mean(np.linalg.norm(diff, axis=0) ** 2)
    den = np.mean(np.linalg.norm(y, axis=0) ** 2)
    vaf = 1 - num/den
    vaf = max(0, vaf*100)

    """# new method
    num = 0
    den = 0
    for k in range(y.shape[-1]):
        norm2_1 = (np.linalg.norm(y[:, k] - yhat[:, k])) ** 2
        num = num + norm2_1
        norm2_2 = (np.linalg.norm(y[:, k])) ** 2
        den = den + norm2_2
    vaf = max(0, (1 - num / den) * 100)
    """

    # print output
    if doprint:
        print('VAF = {:.3f}%'.format(vaf))

    return vaf


# computes the RMSE for all outputs
def compute_rmse(y, yhat, doprint=False):
    # get sizes from data
    num_outputs = y.shape[1]

    # reshape to ydim x -1
    y = y.transpose(1, 0, 2)
    y = y.reshape(num_outputs, -1)
    yhat = yhat.transpose(1, 0, 2)
    yhat = yhat.reshape(num_outputs, -1)

    rmse = np.zeros([num_outputs])
    for i in range(num_outputs):
        rmse[i] = np.sqrt(((yhat[i, :] - y[i, :]) ** 2).mean())

    # print output
    if doprint:
        for i in range(num_outputs):
            print('RMSE y{} = {:.3f}'.format(i + 1, rmse[i]))

    return rmse


# computes the marginal likelihood of all outputs
def compute_marginalLikelihood(y, yhat_mu, yhat_sigma, doprint=False):
    # to torch
    y = torch.tensor(y, dtype=torch.double)
    yhat_mu = torch.tensor(yhat_mu, dtype=torch.double)
    yhat_sigma = torch.tensor(yhat_sigma, dtype=torch.double)

    # number of batches
    num_batches = y.shape[0]
    num_points = np.prod(y.shape)

    # get predictive distribution
    pred_dist = tdist.Normal(yhat_mu, yhat_sigma)

    # get marginal likelihood
    marg_likelihood = torch.mean(pred_dist.log_prob(y))
    # to numpy
    marg_likelihood = marg_likelihood.numpy()

    # print output
    if doprint:
        print('Marginal Likelihood / point = {:.3f}'.format(marg_likelihood))

    return marg_likelihood

# %% plots the resulting time sequence
def plot_time_sequence_uncertainty(data_y_true, data_y_sample, label_y, options, path_general, file_name_general,
                                   batch_show, x_limit_show):
    # storage path
    file_name = file_name_general + '_timeEval'
    path = path_general + 'timeEval/'

    # get number of outputs
    num_outputs = data_y_sample[-1].shape[1]

    # get number of columns
    num_cols = 2

    # initialize figure
    figs, axes =plt.subplots(num_outputs, 2, figsize=(5 * num_cols, 5 * num_outputs))

    # plot outputs
    for j in range(0, num_outputs):
        # output yk

        if len(data_y_true) == 1:  # plot samples
            axes[j,0].plot(data_y_true[0][batch_show, j, :].squeeze(), label='y_{}(k) {}'.format(j + 1, label_y[0]))
        else:  # plot true mu /pm 3sigma
            length = len(data_y_true[0][batch_show, j, :])
            x = np.linspace(0, length - 1, length)
            mu = data_y_true[0][batch_show, j, :].squeeze()
            std = data_y_true[1][batch_show, j, :].squeeze()
            # plot mean
            axes[j,0].plot(mu, label='y_{}(k) {}'.format(j + 1, label_y[0]))
            # plot 3std around
            axes[j,0].fill_between(x, mu, mu + 3 * std, alpha=0.3, facecolor='b')
            axes[j,0].fill_between(x, mu, mu - 3 * std, alpha=0.3, facecolor='b')

        # plot samples mu \pm 3sigma
        length = len(data_y_sample[0][batch_show, j, :])
        x = np.linspace(0, length - 1, length)
        mu = data_y_sample[0][batch_show, j, :].squeeze()
        std = data_y_sample[1][batch_show, j, :].squeeze()

        # plot mean
        axes[j,0].plot(mu, label='y_{}(k) {}'.format(j + 1, label_y[1]))
        # plot 3std around
        axes[j,0].fill_between(x, mu, mu + 3 * std, alpha=0.3, facecolor='r')
        axes[j,0].fill_between(x, mu, mu - 3 * std, alpha=0.3, facecolor='r')

        # plot settings
        axes[j,0].set_title('Output $y_{}(k)$, {} with (h,z,n)=({},{},{})'.format((j + 1),
                                                                        options['dataset'],
                                                                        options['h_dim'],
                                                                        options['z_dim'],
                                                                        options['n_layers']))
        axes[j,0].set_ylabel('$y_{}(k)$'.format(j + 1))
        axes[j,0].set_xlabel('time steps $k$')
        axes[j,0].legend()
        axes[j,0].set_xlim(x_limit_show)
        ###################################
        axes[j,1].scatter(data_y_true[0][batch_show, j, :].squeeze(), data_y_sample[0][batch_show, j, :].squeeze())
        axes[j,1].set_xlabel("true test state $y_{}$".format(j+1))
        axes[j,1].set_ylabel("predict state $y_{}$".format(j+1))
        asp = np.diff(axes[j,1].get_xlim())[0] / np.diff(axes[j,1].get_ylim())[0]
        axes[j,1].set_aspect(asp)
        xmin,xmax = axes[j,1].get_xlim()
        ymin,ymax = axes[j,1].get_ylim()
        minimum_r= min(xmin, ymin)
        maximum_r =max(xmax, ymax)
        diag_line, = axes[j,1].plot(np.arange(minimum_r, maximum_r, 0.02), np.arange(minimum_r, maximum_r, 0.02), ls="--", c=".3")
        figs.tight_layout()

    # save figure
    if options['savefig']:
        # check if path exists and create otherwise
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + file_name + '.png', format='png')
    # plot model
    if options['showfig']:
        plt.show()
    plt.close(1)



def get_n_params(model_to_eval):

    return sum(p.numel() for p in model_to_eval.parameters() if p.requires_grad)

# %% plot and save the loss curve
def plot_losscurve(df, options, path_general, file_name_general, removedata=True):
    # only if df has values
    if 'all_losses' in df:
        # storage path
        file_name = file_name_general + '_loss'
        path = path_general + '/loss/'

        # get data to plot loss curve
        all_losses = df['all_losses']
        all_vlosses = df['all_vlosses']
        time_el = df['train_time']

        # plot loss curve
        plt.figure(1, figsize=(5, 5))
        xval = np.linspace(0, options['test_every'] * (len(all_losses) - 1), len(all_losses))
        plt.plot(xval, all_losses, label='Training set')
        plt.plot(xval, all_vlosses, label='Validation set')  # loss_test_store_idx,
        plt.xlabel('Number Epochs in {:2.0f}:{:2.0f} [min:sec]'.format(time_el // 60,
                                                                       time_el - 60 * (time_el // 60)))
        plt.ylabel('Loss')
        plt.title('Loss  with (h,z,n)=({},{},{})'.format(options['h_dim'],
                                                         options['z_dim'],
                                                         options['n_layers']))
        plt.legend()
        plt.yscale('log')
        # save model
        if options['savefig']:
            # check if path exists and create otherwise
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + file_name + '.png', format='png')
        # show the model
        if options['showfig']:
            plt.show()
        plt.close(1)

        # delete loss value matrices from dictionary
        if removedata:
            del df['all_losses']
            del df['all_vlosses']

    return df

    def normalize_timeseries(timeseries):
        #3D tensor of shape (n_samples, timesteps, n_features) use the following:
        (timeseries-timeseries.min(dim=2))/(timeseries.max(dim=2)-timeseries.min(dim=2))
        return timeseries
    
def _strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict
