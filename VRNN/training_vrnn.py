import pandas as pd
import numpy as np
import datetime 
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
from .Normalization import Normalizer1D, compute_normalizer
from .main import ModelState, DynamicModel, VRNN_GMM
from .vrnn_utilities import *
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import argparse

parser = argparse.ArgumentParser(description='the vrnn transition model (Chung et al. 2016) conditioned on the context, distributed data parallel ')
parser.add_argument('--lr', default=0.0002, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', choices=['gloo', 'nccl'], default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--seed', type=int, default=1234,  help='seed of the experiment')
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:9000"


""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
           dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
           param.grad.data /= size
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            dist.all_reduce(m.running_mean.data, op=dist.reduce_op.SUM)
            m.running_mean.data /= size
            dist.all_reduce(m.running_var.data, op=dist.reduce_op.SUM)
            m.running_var.data /= size

def run_train(modelstate, loader_train, loader_valid, device, dataframe, path_general, file_name_general, lr, max_epochs, train_rank, train_sampler, batch_size):
    train_options={'clip':2,
                   'print_every':2,
                   'test_every':50,
                   'batch_size':batch_size,
                   'n_epochs':max_epochs,
                   'lr_scheduler_nstart':10,#earning rate scheduler start epoch
                   'lr_scheduler_nepochs':5,#check learning rater after
                   'lr_scheduler_factor':10,#adapt learning rate by
                   'min_lr':1e-6,#minimal learning rate
                   'init_lr':lr,#initial learning rate
                   }
    
    
    writer = SummaryWriter(log_dir=os.path.join(path_general, "writer"))
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
                vloss_, d_loss, hidden, real_feature, fake_feature = modelstate.model(u, y)

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
            
        epoch_start = time.time()
        for i, (u, y) in enumerate(loader_train):
            batch_size=u.shape[0]
            start = time.time()
            u = u.to(device)#torch.Size([B, D_in, T])
            y = y.to(device)

            # set the optimizer
                        # set the optimizer
            modelstate.optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float32) and torch.backends.cudnn.flags(enabled=False):
                     loss_, disc_loss, hidden, real, fake = modelstate.model(u, y)
                
                with torch.backends.cudnn.flags(enabled=False):
                     gradient_penalty = modelstate.model.module.wgan_gp_reg(real, fake)
                discriminator_loss = disc_loss + gradient_penalty
                loss_ += discriminator_loss
                scaled_grad_params = torch.autograd.grad(outputs = scaler.scale(loss_),
                                                        inputs   = modelstate.model.parameters(),
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        allow_unused=True #Whether to allow differentiation of unused parameters.
                                                        )

                """
                #Cut NaN value
                for submodule in modelstate.model.modules():
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
                scaler.scale(loss_).backward(retain_graph=True, inputs=list( modelstate.model.parameters()))

                torch.nn.utils.clip_grad_norm_(modelstate.model.parameters(),  max_norm=train_options['clip'], error_if_nonfinite =False)
                # Unscales gradients and calls
                # or skips optimizer.step()

                average_gradients(modelstate.model)
                scaler.step(modelstate.optimizer)
                # Updates the scale for next iteration
                scaler.update()
                batch_time = time.time() - start

                elapse_time = time.time() - epoch_start
                elapse_time = datetime.timedelta(seconds=elapse_time)

                

                print("From Rank: {}, Epoch: {}, Training time {}".format(dist.get_rank(), epoch, elapse_time))

                """
                #test for invalid/NaN values in different layers
                for name, param in modelstate.model.named_parameters():
                    if torch.is_tensor(param.grad):
                        print(name, torch.isfinite(param.grad).all())
                """
            else:
                # forward pass over model
                loss_, disc_loss, hidden, real, fake = modelstate.model(u, y)
                
                with torch.backends.cudnn.flags(enabled=False):
                     gradient_penalty = modelstate.model.module.wgan_gp_reg(real, fake)
                discriminator_loss = disc_loss+ gradient_penalty
                loss_ += discriminator_loss
                # NN optimization
                loss_.backward()
                

                ### GRADIENT CLIPPING
                #
                torch.nn.utils.clip_grad_norm_(modelstate.model.parameters(), train_options['clip'])
                
                modelstate.optimizer.step()

            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_loss += loss_.item() 

            # output to console
            if i % train_options['print_every'] == 0:
                print(
                    'Train Epoch: [{:5d}/{:5d}], Batch [{:6d}/{:6d} ({:3.0f}%)]\tLearning rate: {:.2e}\tTotal Loss: {:.3f}\t Discriminator Loss: {:.4f}'.format(
                        epoch, train_options['n_epochs'], (i + 1), len(loader_train),
                        100. * (i + 1) / len(loader_train), lr, total_loss / total_points, discriminator_loss.item()))  # total_batches
        
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
        progress_bar = tqdm(initial=0, leave=True, total=train_options['n_epochs'], desc=process_desc.format(0, 0, 0), position=0, disable=dist.get_rank()!=0)
                
        for epoch in range(0, train_options['n_epochs'] + 1):
            print('\n[%d]Epoch: %d' % (train_rank, epoch))
            train_sampler.set_epoch(epoch)
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
                    path = path_general + 'saved_model/'
                    file_name = file_name_general + '_bestModel.ckpt'
                    if train_rank==0:
                       # All processes should see same parameters as they all start from same
                       # random parameters and gradients are synchronized in backward passes.
                       # Therefore, saving it in one process is sufficient.
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
        dist.destroy_process_group()
        
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

def run_test(seed, nu, ny, seq_len, loaders, df, device, path_general, file_name_general, batch_size, test_rank, **kwargs):
    # switch to cpu computations for testing
    # options['device'] = 'cpu'

    # %% load model

    # Compute normalizers (here just used for initialization, real values loaded below)
    normalizer_input, normalizer_output = compute_normalizer(loaders)


    # Define model

    modelstate = ModelState(seed=seed,
                            nu=nu,
                            ny=ny,
                            sequence_length=seq_len,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output #
                           )
    modelstate.model.to(device)

    # load model
    model_path = path_general + 'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        pass
    file_name = file_name_general + '_bestModel.ckpt'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % test_rank}
    epoch, vloss = modelstate.load_model(model_path, file_name, map_location='cuda:0')
    print('Best Loaded Train Epoch: {:5d} \tVal Loss: {:.3f}'.format(epoch,  vloss))
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
             'batch_size':batch_size,
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
        u_test[u_test != u_test] = 0 #change nan values to zero
        y_sample, y_sample_mu, y_sample_sigma, hidden = modelstate.model.generate(u_test)

        # convert to cpu and to numpy for evaluation
        # samples data
        y_sample_mu = y_sample_mu.cpu().detach().numpy()
        y_sample_sigma = y_sample_sigma.cpu().detach().numpy()
        # test data
        y_test = y_test.cpu().detach().numpy()
        y_sample = y_sample.cpu().detach().numpy()

    # get noisy test data for narendra_li
    yshape = y_test.shape
    y_test = np.where(np.isnan(y_test), 0, y_test) #change nan values to zero
    y_test_noisy = y_test + np.sqrt(0.1) * np.random.randn(yshape[0], yshape[1], yshape[2])


    # %% plot resulting predictions
    # for narendra_li problem show test data mean pm 3sigma as well
    data_y_true = [y_test, np.sqrt(0.1) * np.ones_like(y_test)]
    data_y_sample = [y_sample_mu, y_sample_sigma]
    label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']

    temp = 250

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


def main():
    print("Starting...")
    path = os.getcwd()
    parentfolder = os.path.dirname(path)
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    os.environ["NCCL_DEBUG"] = "INFO"
    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    print("echo GPUs per node: {}".format(torch.cuda.device_count()))
    print("local ID: ",os.environ.get("SLURM_LOCALID")," node ID: ", os.environ.get("SLURM_NODEID"), "number of tasks: ", os.environ.get("SLURM_NTASKS"))
    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
    print('cuda visible: ',os.environ.get('CUDA_VISIBLE_DEVICES'))
    proc_id = int(os.environ.get("SLURM_PROCID"))
    job_id = int(os.environ.get("SLURM_JOBID"))
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))
    current_device = local_rank

    torch.cuda.set_device(current_device)
    device = torch.device('cuda', local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))
    
    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    print(f"init_method = {args.init_method}")
    #init the process group
    dist.init_process_group(backend=args.dist_backend, 
                            init_method=args.init_method, 
                            world_size=args.world_size, 
                            rank=rank,
                            timeout=datetime.timedelta(0, 240) # 20s connection timeout
                            )
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
    print("echo final check; ngpus_per_node={},local_rank={},rank={},available_gpus={},current_device={}"
              .format(ngpus_per_node,local_rank,rank,available_gpus,current_device))

    # get saving path
    input_filename = "transit_data/gym_transition_inputs.pt"
    target_filename = "transit_data/gym_transition_outputs.pt"
    filepath = os.getcwd()
    parentfolder = os.path.dirname(filepath)
    #read input data
    variable_episodes = torch.load(os.path.join( parentfolder, input_filename))
    final_next_state  = torch.load(os.path.join( parentfolder, target_filename))

    print(f" shape of input data state and action: {variable_episodes.shape}")
    print(f" shape of output data next state and reward: {final_next_state.shape}")
    path_general = parentfolder + '/sac/log/'
    if not os.path.exists(path_general):
        os.makedirs(path_general)
    else:
        pass
    file_name_general='Mujoco'
    
    
    
    # get saving file names
    file_general_path = parentfolder + '/sac/data/'
    if not os.path.exists(file_general_path):
       os.makedirs(file_general_path)
    else:
       pass
    #https://github.com/Abishekpras/vrnn/blob/104b532e862620f9043421e73b98d38653f6b73b/train.py#L71
    seed = args.seed

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

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    print(f" size of training input {x_train.shape}")
    print(f" size of test input {x_test.shape}, validation data size {x_val.shape}")
    print(f" training output {y_train.shape}, test output {y_test.shape}")
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y, validation_x, validation_y = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda(), x_val.cuda(), y_val.cuda()


    u_dim = variable_episodes.shape[-1]
    y_dim = final_next_state.shape[-1]
    seq_len = variable_episodes.shape[1]


    batch_size = args.batch_size
    train_dataset = TensorDataset(train_x, train_y)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank,
                                                                    shuffle=True,
                                                                    )
    #we don't shuffle because it is a time series data
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size , 
                              shuffle=False, 
                              num_workers=args.num_workers, 
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              generator=g,)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size , shuffle=True)


    valid_dataset = TensorDataset(validation_x, validation_y)
    valid_loader = DataLoader(valid_dataset, batch_size=1 , shuffle=True)
    

    normalizer_input, normalizer_output = compute_normalizer(train_loader)

    # Define model
    
    modelstate = ModelState(seed=seed,
                            nu=u_dim,
                            ny=y_dim,
                            sequence_length = seq_len,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output
                            )
    modelstate.model.cuda()
    #for DDP used this instruction: https://docs.alliancecan.ca/wiki/PyTorch 
    modelstate.model = torch.nn.parallel.DistributedDataParallel(modelstate.model, find_unused_parameters=True, device_ids=[current_device])
    print('passed distributed data parallel call')
    df={}# allocation
    # train the model
    file_name_general= file_name_general+'_VRNN_h{}_z{}_n{}'.format(modelstate.h_dim, modelstate.z_dim, modelstate.n_layers)
         
    df = run_train(modelstate=modelstate,
                   loader_train=train_loader,
                   loader_valid=valid_loader,
                   device=device,
                   dataframe=df,
                   path_general=path_general,
                   file_name_general=file_name_general, 
                   lr=args.lr,
                   max_epochs=args.max_epochs,
                   train_rank=rank,
                   train_sampler=train_sampler,
                   batch_size=args.batch_size
                   )
    
    df = run_test(seed = seed,
                  nu   = u_dim, 
                  ny   = y_dim,
                  seq_len = seq_len, 
                  loaders = test_loader, 
                  df = df,
                  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                  path_general = path_general, 
                  file_name_general = file_name_general, 
                  batch_size = args.batch_size,
                  test_rank = rank)
    
    df = pd.DataFrame(df)
    # save data
    file_name = file_general_path + '_VRNN_GMM_GYM_TEST.csv'

    df.to_csv(file_name)
    
if __name__ == "__main__":
   main()
