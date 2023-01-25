import pandas as pd
import numpy as np

import math
import time
import logging
import os, sys
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
import torch.distributions as tdist
from torch.distributions import MultivariateNormal, OneHotCategorical
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.init as init
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from Normalization import Normalizer1D, compute_normalizer
from main import ModelState, DynamicModel, VRNN_GMM
from vrnn_utilities import *
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(modelstate, loader_train, loader_valid, device, dataframe, path_general, file_name_general):
    train_options={'clip':2,
                   'print_every':2,
                   'test_every':50,
                   'batch_size':30,
                   'n_epochs':400,
                   'lr_scheduler_nstart':10,#earning rate scheduler start epoch
                   'lr_scheduler_nepochs':5,#check learning rater after
                   'lr_scheduler_factor':10,#adapt learning rate by
                   'min_lr':1e-6,#minimal learning rate
                   'init_lr':5e-4,#initial learning rate
                   }
    path = os.getcwd()
    parentfolder = os.path.dirname(path)
    
    writer = SummaryWriter(log_dir=os.path.join(parentfolder, "writer"))
    def validate(loader):
        modelstate.model.eval()
        total_vloss = 0
        total_batches = 0
        total_points = 0
        with torch.no_grad():
            for i, (u, y) in enumerate(loader):
                u = u.to(device)
                y = y.to(device)
                # forward pass over model
                with torch.no_grad():
                     vloss_ = modelstate.model(u, y)

                total_batches += u.size()[0]
                total_points += np.prod(u.shape)
                total_vloss += vloss_.item()

        return total_vloss / total_points  # total_batches

    def train(epoch):
        # model in training mode
        modelstate.model.train()
        # initialization
        total_loss = 0
        total_batches = 0
        total_points = 0
        if torch.cuda.is_available():
            #https://pytorch.org/docs/stable/notes/amp_examples.html
            scaler = torch.cuda.amp.GradScaler()

        for i, (u, y) in enumerate(loader_train):
            batch_size=u.shape[0]
            u = u.to(device)#torch.Size([B, D_in, T])
            y = y.to(device)

            # set the optimizer
                        # set the optimizer
            modelstate.optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float32) and torch.backends.cudnn.flags(enabled=False):
                    loss_ = modelstate.model(u, y)
                diff_params = [p for p in modelstate.model.m.parameters() if p.requires_grad]
                scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss_),
                                                        inputs=diff_params,
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        allow_unused=True #Whether to allow differentiation of unused parameters.
                                                        )

                """
                #Cut NaN value
                for submodule in modelstate.model.m.modules():
                    submodule.register_forward_hook(nan_hook)
                """
                inv_scale = 1./scaler.get_scale()

                grad_params = [ p * inv_scale if p is not None and not torch.isnan(p).any() else torch.tensor(0, device=device, dtype=torch.float32) for p in scaled_grad_params ]
                #grad_params = [p * inv_scale for p in scaled_grad_params]
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    #grad_norm = torch.tensor(0, device=grad_params[0].device, dtype=grad_params[0].dtype)
                    grad_norm = 0
                    for grad in grad_params:
                        grad_norm += grad.pow(2).sum()
                        grad_norm = grad_norm**0.5
                    # Compute the L2 Norm as penalty and add that to loss
                    loss_ = loss_ + grad_norm


                # Scales the loss, and calls backward()
                # to create scaled gradients
                assert not torch.isnan(loss_)
                scaler.scale(loss_).backward(retain_graph=True, inputs=list(modelstate.model.m.parameters()))

                torch.nn.utils.clip_grad_norm_(modelstate.model.m.parameters(),  max_norm=train_options['clip'], error_if_nonfinite =False)
                # Unscales gradients and calls
                # or skips optimizer.step()


                scaler.step(modelstate.optimizer)

                # Updates the scale for next iteration
                scaler.update()
                """
                #test for invalid/NaN values in different layers
                for name, param in modelstate.model.m.named_parameters():
                    if torch.is_tensor(param.grad):
                        print(name, torch.isfinite(param.grad).all())
                """
            else:
                # forward pass over model
                loss_ = modelstate.model(u, y)
                # NN optimization
                loss_.backward()
                ### GRADIENT CLIPPING
                #
                torch.nn.utils.clip_grad_norm_(modelstate.model.m.parameters(), train_options['clip'])

                modelstate.optimizer.step()


            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_loss += loss_.item()

            # output to console
            if i % train_options['print_every'] == 0:
                print(
                    'Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}'.format(
                        epoch, train_options['n_epochs'], (i + 1), len(loader_train),
                        100. * (i + 1) / len(loader_train), lr, total_loss / total_points))  # total_batches
                writer.add_scalar('data/total_loss', total_loss / total_points, i + epoch*len(loader_train))
        return total_loss / total_points

    try:

        modelstate.model.train()
        # Train
        vloss = validate(loader_valid)
        all_losses = []
        all_vlosses = []
        best_vloss = vloss

        print("Start training the model ....")
        start_time = time.time()

        # Extract initial learning rate
        lr = train_options['init_lr']

        # output parameter
        best_epoch = 0
        process_desc = "Train-loss: {:2.3e}; Valid-loss: {:2.3e}; LR: {:2.3e}"
        progress_bar = tqdm(initial=0, leave=True, total=train_options['n_epochs'], desc=process_desc.format(0, 0, 0), position=0)
        
        
        for epoch in range(0, train_options['n_epochs'] + 1):
            # Train and validate
            train(epoch)  # model, train_options, loader_train, optimizer, epoch, lr)
            # validate every n epochs
            if epoch % train_options['test_every'] == 0:
                vloss = validate(loader_valid)
                loss = validate(loader_train)
                # Save losses
                all_losses += [loss]
                all_vlosses += [vloss]

                if vloss < best_vloss:  # epoch == train_options.n_epochs:  #
                    best_vloss = vloss
                    # save model
                    path = path_general + 'model/'
                    file_name = file_name_general + '_bestModel.ckpt'
                    modelstate.save_model(epoch, vloss, time.process_time() - start_time, path, file_name)
                    # torch.save(model.state_dict(), path + file_name)
                    best_epoch = epoch

                # Print validation results
                print('Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tLoss: {:.3f}'
                      '\tVal Loss: {:.3f}'.format(epoch, train_options['n_epochs'], len(loader_train), len(loader_train),
                                                  100., lr, loss, vloss))

                # lr scheduler
                if epoch >= train_options['lr_scheduler_nstart']:
                    if len(all_vlosses) > train_options['lr_scheduler_nepochs'] and \
                            vloss >= max(all_vlosses[int(-train_options['lr_scheduler_nepochs'] - 1):-1]):
                        # reduce learning rate
                        lr = lr / train_options['lr_scheduler_factor']
                        # adapt new learning rate in the optimizer
                        for param_group in modelstate.optimizer.param_groups:
                            param_group['lr'] = lr
                        #print('\nLearning rate adapted! New learning rate {:.3e}\n'.format(lr))
                        message = 'Learning rate adapted in epoch {} with valid loss {:2.6e}. New learning rate {:.3e}.'
                        tqdm.write(message.format(epoch, vloss, lr))
                
            progress_bar.desc = process_desc.format(loss, vloss, lr)
            progress_bar.update(1)
            # Early stoping condition
            if lr < train_options['min_lr']:
                break
        progress_bar.close()
        
    except KeyboardInterrupt:
        tqdm.write('\n')
        tqdm.write('-' * 89)
        tqdm.write('Exiting from training early.......')
        # modelstate.save_model(epoch, vloss, time.clock() - start_time, logdir, 'interrupted_model.pt')
        tqdm.writet('-' * 89)

    # print best saved epoch model
    # print('\nBest model from epoch {} saved.'.format(best_epoch))

    # print time of learning
    time_el = time.time() - start_time
    # print('\nTotal learning time: {:2.0f}:{:2.0f} [min:sec]'.format(time_el // 60, time_el - 60 * (time_el // 60)))

    # save data in dictionary
    train_dict = {'all_losses': all_losses,
                  'all_vlosses': all_vlosses,
                  'best_epoch': best_epoch,
                  'total_epoch': epoch,
                  'train_time': time_el}
    # overall options
    dataframe.update(train_dict)

    return dataframe

def run_test(seed, nu, ny, loaders, df, device, path_general, file_name_general, **kwargs):
    # switch to cpu computations for testing
    # options['device'] = 'cpu'

    # %% load model

    # Compute normalizers (here just used for initialization, real values loaded below)
    normalizer_input, normalizer_output = compute_normalizer(loaders)


    # Define model
    modelstate = ModelState(seed=seed,
                            nu=nu,
                            ny=ny,
                            #normalizer_input=normalizer_input,
                            #normalizer_output=normalizer_output #
                           )
    modelstate.model.to(device)

    # load model
    path = path_general + 'model/'
    file_name = file_name_general + '_bestModel.ckpt'
    modelstate.load_model(path, file_name)
    modelstate.model.to(device)
    options={'h_dim':modelstate.h_dim,
             'z_dim':modelstate.z_dim,
             'n_layers':modelstate.n_layers,
             'n_mixtures':modelstate.n_mixtures,
             'dataset':'Hopper',
             'test_every':50,
             'showfig':'True',
             'savefig':'True',
             'seq_len_train':loaders.dataset[0][-1].shape[-1],
             'batch_size':30,
             'lr_scheduler_nepochs':5,
             'lr_scheduler_factor':10}
    # %% plot and save the loss curve
    plot_losscurve(df, options, path_general, file_name_general)

    # %% others

    if bool(kwargs):
        file_name_add = kwargs['file_name_add']
    else:
        # Default option
        file_name_add = ''
    file_name_general = file_name_add + file_name_general

    # get the number of model parameters
    num_model_param = get_n_params(modelstate.model)
    print('Model parameters: {}'.format(num_model_param))

    # %% RUN PERFORMANCE EVAL
    # %%

    # %% sample from the model
    for i, (u_test, y_test) in enumerate(loaders):
        # getting output distribution parameter only implemented for selected models
        u_test = u_test.to(device)

        y_sample, y_sample_mu, y_sample_sigma = modelstate.model.generate(u_test)

        # convert to cpu and to numpy for evaluation
        # samples data
        y_sample_mu = y_sample_mu.cpu().detach().numpy()
        y_sample_sigma = y_sample_sigma.cpu().detach().numpy()
        # test data
        y_test = y_test.cpu().detach().numpy()
        y_sample = y_sample.cpu().detach().numpy()

    # get noisy test data for narendra_li
    yshape = y_test.shape
    y_test_noisy = y_test + np.sqrt(0.1) * np.random.randn(yshape[0], yshape[1], yshape[2])


    # %% plot resulting predictions
    # for narendra_li problem show test data mean pm 3sigma as well
    data_y_true = [y_test, np.sqrt(0.1) * np.ones_like(y_test)]
    data_y_sample = [y_sample_mu, y_sample_sigma]
    label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']

    temp = 200

    plot_time_sequence_uncertainty(data_y_true,
                                   data_y_sample,
                                   label_y,
                                   options,
                                   batch_show=0,
                                   x_limit_show=[0, temp],
                                   path_general=path_general,
                                   file_name_general=file_name_general)

    # %% compute performance values

    # compute marginal likelihood (same as for predictive distribution loss in training)
    marginal_likeli = compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)

    # compute VAF
    vaf = compute_vaf(y_test_noisy, y_sample_mu, doprint=True)

    # compute RMSE
    rmse = compute_rmse(y_test_noisy, y_sample_mu, doprint=True)

    # %% Collect data

    # options_dict
    options_dict = {'h_dim': options['h_dim'],
                    'z_dim': options['z_dim'],
                    'n_layers': options['n_layers'],
                    'seq_len_train': options['seq_len_train'],
                    'batch_size': options['batch_size'],
                    'lr_scheduler_nepochs': options['lr_scheduler_nepochs'],
                    'lr_scheduler_factor': options['lr_scheduler_factor'],
                    'model_param': num_model_param, }
    # test_dict
    test_dict = {'marginal_likeli': marginal_likeli,
                 'vaf': vaf,
                 'rmse': rmse}
    # dataframe
    df.update(options_dict)
    df.update(test_dict)

    return df

if __name__ == "__main__":
    # get saving path

    input_filename = "transit_data/gym_transition_inputs.pt"
    target_filename = "transit_data/gym_transition_outputs.pt"
    filepath = os.getcwd()
    parentfolder = os.path.dirname(filepath)
    variable_episodes = torch.load(os.path.join( parentfolder, input_filename))
    final_next_state  = torch.load(os.path.join( parentfolder, target_filename))

    print(f" shape of input data (s_t, a_t): {variable_episodes.shape}")
    print(f" shape of output data (s_t+1, r_t): {final_next_state.shape}")
    path_general = parentfolder + '/sac/log/'
    if not os.path.exists(path_general):
        os.makedirs(path_general)
    else:
        pass
    file_name_general='Mujoco'

    IMG_FOLDER=parentfolder +'/sac/plots'
    #LOG_FILENAME = '/content/gdrive/My Drive/sac/tmp/logs/vrnn_{}.log'.format(time.strftime("%Y%m%d-%H%M%S"))
    #logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    logger = logging.getLogger('my_logger')
    logging.basicConfig(
      level=logging.DEBUG # allow DEBUG level messages to pass through the logger
    )


    logging.getLogger(__name__).addHandler(logging.StreamHandler(sys.stdout))
    logger.info('module initialized')

    # get saving file names
    file_name_general = parentfolder + '/sac/data/'
    if not os.path.exists(file_name_general):
       os.makedirs(file_name_general)
    else:
       pass
    #https://github.com/Abishekpras/vrnn/blob/104b532e862620f9043421e73b98d38653f6b73b/train.py#L71
    seed = 1234

    plt.ion()
    train_ratio = 0.34
    validation_ratio = 0.33
    test_ratio = 0.33
    assert torch.isfinite(variable_episodes).any()
    assert torch.isfinite(final_next_state).any()
    assert not torch.isnan(final_next_state).any()
    assert not torch.isnan(variable_episodes).any()
    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(variable_episodes.permute(0,2,1), final_next_state.permute(0,2,1), test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    #scale the dataset



    print(f" size of training input {x_train.shape}")
    print(f" size of test input {x_test.shape}, validation data size {x_val.shape}")
    print(f" training output {y_train.shape}, test output {y_test.shape}")
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y, validation_x, validation_y = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda(), x_val.cuda(), y_val.cuda()


    u_dim = variable_episodes.shape[-1]
    y_dim = final_next_state.shape[-1]


    batch_size = 30
    train_dataset = TensorDataset(train_x, train_y)
    #we don't shuffle because it is a time series data
    train_loader = DataLoader(train_dataset, batch_size=batch_size , shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size , shuffle=True)


    valid_dataset = TensorDataset(validation_x, validation_y)
    valid_loader = DataLoader(valid_dataset, batch_size=1 , shuffle=True)
    

    normalizer_input, normalizer_output = compute_normalizer(train_loader)

    # Define model
    modelstate = ModelState(seed=seed,
                            nu=u_dim,
                            ny=y_dim,
                            #normalizer_input=normalizer_input,
                            #normalizer_output=normalizer_output
                            )
    modelstate.model.to(device)
    df={}# allocation
    # train the model
    file_name_general= file_name_general+'VRNN_h{}_z{}_n{}'.format(modelstate.h_dim, modelstate.z_dim, modelstate.n_layers)

    df = run_train(modelstate=modelstate,
                   loader_train=train_loader,
                   loader_valid=valid_loader,
                   device=device,
                   dataframe=df,
                   path_general=path_general,
                   file_name_general=file_name_general, )

    df = run_test(seed, u_dim, y_dim, test_loader, df,torch.device('cuda' if torch.cuda.is_available() else 'cpu'), path_general, file_name_general)
    df = pd.DataFrame(df)

    # save data
    file_name = file_name_general + 'VRNN_GMM_GYM_TEST.csv'

    df.to_csv(file_name)
