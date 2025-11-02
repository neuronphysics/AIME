import datetime
import time
import torch.utils.data
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from contextlib import nullcontext
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VRNN.Normalization import Normalizer1D, compute_normalizer
from VRNN.main import ModelState, DynamicModel, VRNN_GMM
from VRNN.vrnn_utilities import *
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch.utils.data.distributed
import torch.distributed as dist
import random
import argparse
import pandas as pd
from advanced_metrics import compute_kl_divergence_per_dim, compute_confidence_adjusted_error_per_dim, compute_log_likelihood_per_dim, plot_dimension_metrics, plot_calibration_curves


parser = argparse.ArgumentParser(
    description='VRNN transition model (Chung et al. 2016) conditioned on the context, with distributed data parallel')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
parser.add_argument('--max_epochs', type=int, default=4, help='Maximum number of epochs to train')
parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='URL for distributed training')
parser.add_argument('--dist-backend', choices=['gloo', 'nccl'], default='gloo', type=str, help='Distributed backend')
parser.add_argument('--world_size', default=1, type=int, help='Number of processes for distributed training')
parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
parser.add_argument('--vis_backend', choices=['tensorboard', 'wandb', 'both', 'none'], default='tensorboard', 
                    help='Visualization backend: tensorboard, wandb, both, or none')
parser.add_argument('--wandb_project', type=str, default='vrnn-dynamics', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name (username or team name)')
parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated list of tags for wandb run')
parser.add_argument('--h_dim', type=int, default=80, help='Hidden dimension size')
parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension size')
parser.add_argument('--n_layers', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--n_mixtures', type=int, default=8, help='Number of Gaussian mixtures')

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:9000"

""" Gradient averaging for distributed training """


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


def init_visualization(args, rank, model_config):
    """Initialize visualization backends based on command-line args"""
    writers = {}
    
    # Only initialize on the main process (rank 0) to avoid duplicates
    if rank != 0:
        return writers
    
    # Initialize wandb
    if args.vis_backend in ['wandb', 'both']:
        # Create a more descriptive run name
        run_name = f"VRNN_h{model_config['h_dim']}_z{model_config['z_dim']}_n{model_config['n_layers']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert wandb tags to list if provided
        tags = args.wandb_tags.split(',') if args.wandb_tags else None
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config={
                "h_dim": model_config['h_dim'],
                "z_dim": model_config['z_dim'],
                "n_layers": model_config['n_layers'],
                "n_mixtures": model_config['n_mixtures'],
                "sequence_length": model_config['sequence_length'],
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_epochs": args.max_epochs,
                "seed": args.seed,
                "distributed": args.distributed,
                "world_size": args.world_size
            }
        )
        writers['wandb'] = True
    
    # Initialize TensorBoard
    if args.vis_backend in ['tensorboard', 'both']:
        tensorboard_dir = os.path.join(model_config['path_general'], "tensorboard")
        writers['tensorboard'] = SummaryWriter(log_dir=tensorboard_dir)
    
    return writers


def compute_calibration_curve(y, yhat_mu, yhat_sigma, dim_index):
    """Compute calibration curve data for a specific dimension"""
    import scipy.stats as stats
    
    # Confidence levels to evaluate
    confidence_levels = np.linspace(0.1, 0.99, 30)
    
    # Compute valid sequence lengths for each batch item
    seq_len = [np.max(np.where(~np.isnan(y[i, 0, :]))[0]) + 1 if np.any(~np.isnan(y[i, 0, :])) else 0 
               for i in range(y.shape[0])]
    
    # Collect all normalized errors for this dimension
    normalized_errors = []
    for b in range(y.shape[0]):
        for t in range(seq_len[b]):
            if not np.isnan(y[b, dim_index, t]) and yhat_sigma[b, dim_index, t] > 0:
                error = (y[b, dim_index, t] - yhat_mu[b, dim_index, t]) / yhat_sigma[b, dim_index, t]
                normalized_errors.append(error)
    
    if len(normalized_errors) == 0:
        return [np.nan] * len(confidence_levels)
    
    normalized_errors = np.array(normalized_errors)
    
    # Calculate actual coverage for each confidence level
    actual_coverage = []
    for conf_level in confidence_levels:
        # Calculate the z-score for this confidence level
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        
        # Calculate the fraction of errors within this z-score
        coverage = np.mean(np.abs(normalized_errors) <= z_score)
        actual_coverage.append(coverage)
    
    return confidence_levels, actual_coverage, normalized_errors


def log_to_visualization(writers, metrics, step=None, prefix=''):
    """Log metrics to all enabled visualization backends"""
    if not writers:
        return
    
    # Log to TensorBoard
    if 'tensorboard' in writers and writers['tensorboard'] is not None:
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                writers['tensorboard'].add_scalar(f"{prefix}/{name}", value, step)
    
    # Log to wandb
    if 'wandb' in writers and writers['wandb']:
        # Only log if a prefix is provided to avoid confusing wandb
        if prefix:
            wandb_metrics = {f"{prefix}/{name}": value for name, value in metrics.items()}
        else:
            wandb_metrics = metrics
        
        # Add step if provided
        if step is not None:
            wandb_metrics['step'] = step
            
        wandb.log(wandb_metrics)

def visualize_gradient_dynamics(writers, generator_grads, discriminator_grads, epoch):
    """Create detailed gradient dynamics visualization across model components"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Generator gradients across architectural components
    axes[0].plot(generator_grads['encoder'], label='Encoder', linestyle='-')
    axes[0].plot(generator_grads['decoder'], label='Decoder', linestyle='--')
    axes[0].plot(generator_grads['rnn'], label='RNN', linestyle=':')
    axes[0].set_title('Generator Gradient Magnitudes', fontsize=14)
    axes[0].set_ylabel('L2 Norm', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Discriminator gradient trajectory
    axes[1].plot(discriminator_grads, label='Discriminator', color='red')
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title('Discriminator Gradient Magnitudes', fontsize=14)
    axes[1].set_xlabel('Training Iterations', fontsize=12)
    axes[1].set_ylabel('L2 Norm', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    log_figure_to_visualization(writers, fig, 'gradient_dynamics', epoch)
    
def log_figure_to_visualization(writers, figure, name, step=None):
    """Log a matplotlib figure to visualization backends"""
    if not writers:
        return
    
    # Save the figure so we can log it to both backends
    # Create a temporary file for the figure
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        figure_path = tmpfile.name
        plt.savefig(figure_path, format='png', dpi=300)
    
    # Log to TensorBoard
    if 'tensorboard' in writers and writers['tensorboard'] is not None:
        writers['tensorboard'].add_figure(name, figure, step)
    
    # Log to wandb
    if 'wandb' in writers and writers['wandb']:
        wandb.log({name: wandb.Image(figure_path)})
    
    # Clean up the temporary file
    os.remove(figure_path)


def log_calibration_table_to_wandb(confidence_levels, actual_coverage, dimensions, dimension_indices):
    """Log calibration data as an interactive table to wandb"""
    table = wandb.Table(columns=["Confidence Level"] + [f"Dim {i+1}" for i in dimension_indices])
    
    for i in range(len(confidence_levels)):
        row = [confidence_levels[i]] + [actual_coverage[dim][i] for dim in range(dimensions)]
        table.add_data(*row)
    
    wandb.log({"calibration_table": wandb.plot.line(
        table, "Confidence Level", 
        [f"Dim {i+1}" for i in dimension_indices],
        title="Calibration Curves")})


def log_metrics_by_dimension_to_wandb(kl_div, conf_adj_rmse, log_likelihood):
    """Log per-dimension metrics as an interactive chart to wandb"""
    # Get indices of dimensions with valid data
    valid_indices = []
    for i in range(len(kl_div)):
        if not np.isnan(kl_div[i]) and not np.isnan(conf_adj_rmse[i]) and not np.isnan(log_likelihood[i]):
            valid_indices.append(i)
    
    if not valid_indices:
        return  # No valid dimensions to log
    
    table = wandb.Table(columns=["Dimension", "KL Divergence", "Conf-Adj RMSE", "Log-Likelihood"])
    
    for i in valid_indices:
        table.add_data(f"Dim {i+1}", kl_div[i], conf_adj_rmse[i], log_likelihood[i])
    
    wandb.log({"dimension_metrics": wandb.plot.bar(
        table, "Dimension", ["KL Divergence", "Conf-Adj RMSE", "Log-Likelihood"],
        title="Metrics by Dimension")})

def compute_gradient_norm(parameters):
    """Calculate gradient norm for monitoring without adding to loss"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def run_train(modelstate, loader_train, loader_valid, device, dataframe, path_general, file_name_general, lr,
              max_epochs, train_rank, train_sampler, batch_size, writers):
    train_options = {'clip': 2,
                     'print_every': 5,
                     'test_every': 50,
                     'batch_size': batch_size,
                     'n_epochs': max_epochs,
                     'lr_scheduler_nstart': 10,  # learning rate scheduler start epoch
                     'lr_scheduler_nepochs': 5,  # check learning rate after
                     'lr_scheduler_factor': 10,  # adapt learning rate by
                     'min_lr': 1e-6,  # minimal learning rate
                     'init_lr': lr,  # initial learning rate
                     }

    if train_rank == 0:  # Only log from rank 0 to avoid duplication
        try:
            import psutil
            import GPUtil
            track_resources = True
        except ImportError:
            track_resources = False
            print("psutil or GPUtil not available, resource tracking disabled")

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
                vloss_, d_loss, hidden, real_feature, fake_feature, attention_latent = modelstate.model(u, y)

                total_batches += u.size()[0]
                total_points += np.prod(u.shape)
                total_vloss += vloss_.item()

        return total_vloss / total_points  # total_batches

    def train(epoch):
        # model in training mode
        n_critic = 3
        
        modelstate.model.train()
        discriminator_parameters = [p for n, p in modelstate.model.named_parameters() 
                           if 'discriminator' in n]
        generator_parameters = [p for n, p in modelstate.model.named_parameters() 
                       if 'discriminator' not in n]
        # initialization
        total_loss = 0
        total_batches = 0
        total_points = 0
        total_disc_loss = 0
        total_grad_norm = 0
        if torch.cuda.is_available():
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            scaler = torch.cuda.amp.GradScaler()

        epoch_start = time.time()
        for i, (u, y) in enumerate(loader_train):
        
            start = time.time()
            u = u.to(device)  # torch.Size([B, D_in, T])
            y = y.to(device)

            # set the optimizer
            modelstate.optimizer.zero_grad()

            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float32) if torch.cuda.is_available() else nullcontext():
                    vrnn_loss, disc_loss, hidden, real, fake, attention_latent = modelstate.model(u, y)
                # 1. Update Discriminator (multiple iterations for stability)
                for _ in range(n_critic):
                    modelstate.optimizer_disc.zero_grad()
            
                    with torch.backends.cudnn.flags(enabled=False):
                        gradient_penalty = modelstate.model.module.wgan_gp_reg(real, fake)
            
                    # Wasserstein loss with gradient penalty
                    disc_total_loss = disc_loss + gradient_penalty
            
                    if torch.cuda.is_available():
                        scaler.scale(disc_total_loss).backward(retain_graph=True)
                        scaler.unscale_(modelstate.optimizer_disc)
                        torch.nn.utils.clip_grad_norm_(discriminator_parameters, train_options['clip'])
                        scaler.step(modelstate.optimizer_disc)
                        scaler.update()
                    else:
                        disc_total_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(discriminator_parameters, train_options['clip'])
                        modelstate.optimizer_disc.step()

                # 2. Update Generator
                modelstate.optimizer_gen.zero_grad()
        
                # Generator loss combines VRNN reconstruction + KL terms with adversarial component
                gen_loss = vrnn_loss - torch.mean(disc_loss)  # Negative discriminator loss for generator
        
                if torch.cuda.is_available():
                    scaler.scale(gen_loss).backward()
                    scaler.unscale_(modelstate.optimizer_gen)
                    torch.nn.utils.clip_grad_norm_(generator_parameters, train_options['clip'])
                    scaler.step(modelstate.optimizer_gen)
                    scaler.update()
                else:
                    gen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator_parameters, train_options['clip'])
                    modelstate.optimizer_gen.step()
        
                # Compute gradient norm for monitoring only
                grad_norm = compute_gradient_norm(modelstate.model.parameters())
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
                loss_, disc_loss, hidden, real, fake, attention_latent = modelstate.model(u, y)

                with torch.backends.cudnn.flags(enabled=False):
                    gradient_penalty = modelstate.model.module.wgan_gp_reg(real, fake)
                discriminator_loss = disc_loss + gradient_penalty
                loss_ += discriminator_loss
                # NN optimization
                loss_.backward()

                ### GRADIENT CLIPPING
                #
                torch.nn.utils.clip_grad_norm_(modelstate.model.parameters(), train_options['clip'])

                modelstate.optimizer.step()
                
            # Log batch metrics
            if train_rank == 0 and i % train_options['print_every'] == 0:
                global_step = epoch * len(loader_train) + i
                
                # Log training metrics
                metrics = {
                    'total_loss': loss_.item(),
                    'discriminator_loss': discriminator_loss.item(),
                    'gradient_norm': grad_norm.item(),
                    'learning_rate': lr
                }
                
                # Log to visualization backends
                log_to_visualization(writers, metrics, global_step, prefix='training')
                
                # Resource tracking
                if track_resources:
                    resource_metrics = {
                        'cpu_percent': psutil.cpu_percent(),
                        'ram_percent': psutil.virtual_memory().percent
                    }
                    
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        resource_metrics.update({
                            'gpu_utilization': gpus[0].load * 100,
                            'gpu_memory_used': gpus[0].memoryUsed
                        })
                    
                    log_to_visualization(writers, resource_metrics, global_step, prefix='system')
            
            total_batches += u.size()[0]
            total_points += np.prod(u.shape)
            total_loss += loss_.item()
            total_disc_loss += discriminator_loss.item()
            total_grad_norm += grad_norm.item()
            
            # Output to console
            if i % train_options['print_every'] == 0:
                time_per_batch = batch_time
                est_epoch_time = time_per_batch * len(loader_train)
                eta = datetime.timedelta(seconds=est_epoch_time * (train_options['n_epochs'] - epoch))
                print('TRAIN | Epoch: [{:3d}/{:3d}] | Batch: [{:4d}/{:4d} ({:3.0f}%)] | LR: {:.2e} | Loss: {:.4f} | Disc Loss: {:.4f} | ETA: {}'.format(
                        epoch, train_options['n_epochs'], (i + 1), len(loader_train),
                        100. * (i + 1) / len(loader_train), lr, total_loss / total_points,
                        discriminator_loss.item(), eta))
                
        # Log epoch metrics
        if train_rank == 0:
            avg_metrics = {
                'train_loss': total_loss / total_points,
                'disc_loss': total_disc_loss / total_batches,
                'grad_norm': total_grad_norm / len(loader_train)
            }
            log_to_visualization(writers, avg_metrics, epoch, prefix='epoch')
            
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
        process_desc = "Epoch: {}/{} - Train-loss: {:2.3e}; Valid-loss: {:2.3e}; LR: {:2.3e}"
        progress_bar = tqdm(initial=0, leave=True, total=train_options['n_epochs'], 
                            desc=process_desc.format(0, train_options['n_epochs'], 0, 0, 0),
                            position=0, disable=dist.get_rank() != 0)

        for epoch in range(0, train_options['n_epochs'] + 1):
            print('\n' + '='*80)
            print('Starting Epoch: [{:5d}/{:5d}] from Rank: {}'.format(epoch, train_options['n_epochs'], train_rank))
            print('='*80)

            train_sampler.set_epoch(epoch)
            # Train and validate
            train(epoch)  # model, train_options, loader_train, optimizer, epoch, lr)
            
            # Validate every n epochs
            if epoch % train_options['test_every'] == 0:
                vloss = validate(loader_valid)
                loss = validate(loader_train)
                
                # Save losses
                all_losses += [loss]
                all_vlosses += [vloss]

                # Log validation metrics
                if train_rank == 0:
                    val_metrics = {
                        'validation_loss': vloss,
                        'train_eval_loss': loss
                    }
                    log_to_visualization(writers, val_metrics, epoch)

                if vloss < best_vloss:  # epoch == train_options.n_epochs:  #
                    best_vloss = vloss
                    # save model
                    path = path_general + 'saved_model/'
                    file_name = file_name_general + '_bestModel.ckpt'
                    if train_rank == 0:
                        # All processes should see same parameters as they all start from same
                        # random parameters and gradients are synchronized in backward passes.
                        # Therefore, saving it in one process is sufficient.
                        modelstate.save_model(epoch, vloss, time.process_time() - start_time, path, file_name)
                        
                        # Log best model to wandb if enabled
                        if 'wandb' in writers and writers['wandb']:
                            model_path = os.path.join(path, file_name)
                            wandb.save(model_path)
                            wandb.run.summary.update({
                                "best_epoch": epoch,
                                "best_val_loss": vloss
                            })
                    
                    # torch.save(model.state_dict(), path + file_name)
                    best_epoch = epoch

                # Print validation results
                print('\n' + '-'*80)
                print('VALIDATION | Epoch: [{:5d}/{:5d}] - Train Loss: {:.5f}, Validation Loss: {:.5f}, LR: {:.5e}'.format(
                        epoch, train_options['n_epochs'], loss, vloss, lr))
                if vloss < best_vloss:
                    print('NEW BEST MODEL SAVED!')
                print('-'*80)
                
                # Learning rate scheduler
                if epoch >= train_options['lr_scheduler_nstart']:
                    if len(all_vlosses) > train_options['lr_scheduler_nepochs'] and \
                            vloss >= max(all_vlosses[int(-train_options['lr_scheduler_nepochs'] - 1):-1]):
                        # reduce learning rate
                        lr = lr / train_options['lr_scheduler_factor']
                        # adapt new learning rate in the optimizer
                        for param_group in modelstate.optimizer.param_groups:
                            param_group['lr'] = lr
                        # print('\nLearning rate adapted! New learning rate {:.3e}\n'.format(lr))
                        message = 'Learning rate adapted in epoch {} with valid loss {:2.6e}. New learning rate {:.3e}.'
                        tqdm.write(message.format(epoch, vloss, lr))
                        
                        # Log LR change to visualization backends
                        if train_rank == 0:
                            lr_metrics = {'learning_rate': lr}
                            log_to_visualization(writers, lr_metrics, epoch)

            # Update progress bar
            progress_bar.desc = process_desc.format(epoch, train_options['n_epochs'], loss, vloss, lr)
            progress_bar.update(1)
            
            # Early stopping condition
            if lr < train_options['min_lr']:
                break
                
        progress_bar.close()
        
        # Finish wandb run if enabled
        if train_rank == 0 and 'wandb' in writers and writers['wandb']:
            wandb.finish()
            
        dist.destroy_process_group()

    except KeyboardInterrupt:
        tqdm.write('\n')
        tqdm.write('-' * 89)
        tqdm.write('Exiting from training early.......')
        # modelstate.save_model(epoch, vloss, time.process_time() - start_time, logdir, 'interrupted_model.pt')
        tqdm.write('-' * 89)
        
        # Finish wandb run if enabled
        if train_rank == 0 and 'wandb' in writers and writers['wandb']:
            wandb.finish()

    # Print time of learning
    time_el = time.time() - start_time

    # Save data in dictionary
    train_dict = {'all_losses': all_losses,
                  'all_vlosses': all_vlosses,
                  'best_epoch': best_epoch,
                  'total_epoch': epoch,
                  'train_time': time_el}
    # overall options
    dataframe.update(train_dict)

    return dataframe


def run_test(seed, nu, ny, seq_len, loaders, df, device, path_general, file_name_general, batch_size, test_rank,
             writers, **kwargs):
    # Compute normalizers (here just used for initialization, real values loaded below)
    test_x, test_y = loaders.dataset.tensors
    normalizer_input, normalizer_output = compute_normalizer(test_x.transpose(1, 2).cpu(), test_y.transpose(1, 2).cpu())

    # Define model
    modelstate = ModelState(seed=seed,
                            nu=nu,
                            ny=ny,
                            sequence_length=seq_len,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output  #
                            )
    modelstate.model.to(device)

    # Load best model
    model_path = path_general + 'saved_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    file_name = file_name_general + '_bestModel.ckpt'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % test_rank}
    epoch, vloss = modelstate.load_model(model_path, file_name, map_location='cuda:0')
    print('Best Loaded Train Epoch: {:5d} \tVal Loss: {:.3f}'.format(epoch, vloss))
    modelstate.model.to(device)
    
    options = {'h_dim': modelstate.h_dim,
               'z_dim': modelstate.z_dim,
               'n_layers': modelstate.n_layers,
               'n_mixtures': modelstate.n_mixtures,
               'wm_image_replay_buffer': 'Hopper',
               'test_every': 50,
               'showfig': 'True',
               'savefig': 'True',
               'seq_len_train': test_x.shape[-1],
               'batch_size': batch_size,
               'lr_scheduler_nepochs': 5,
               'lr_scheduler_factor': 10}
    
    # Plot and save the loss curve
    plot_losscurve(df, options, path_general, file_name_general)

    # Handle file naming options
    if bool(kwargs):
        file_name_add = kwargs['file_name_add']
    else:
        # Default option
        file_name_add = ''
    file_name_general = file_name_add + file_name_general

    # Get the number of model parameters
    num_model_param = get_n_params(modelstate.model)
    print('Model parameters: {}'.format(num_model_param))

    # Sample from the model
    for i, (u_test, y_test) in enumerate(loaders):
        u_test = u_test.to(device)
        u_test[u_test != u_test] = 0  # change nan values to zero
        y_sample, y_sample_mu, y_sample_sigma, hidden = modelstate.model.generate(u_test)

        # Convert to CPU and to numpy for evaluation
        y_sample_mu = y_sample_mu.cpu().detach().numpy()
        y_sample_sigma = y_sample_sigma.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        y_sample = y_sample.cpu().detach().numpy()

    # Get noisy test data
    yshape = y_test.shape
    y_test = np.where(np.isnan(y_test), 0, y_test)  # change nan values to zero
    y_test_noisy = y_test + np.sqrt(0.1) * np.random.randn(yshape[0], yshape[1], yshape[2])

    # Plot resulting predictions
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

    print("\n--- Computing Performance Metrics ---")
    
    # Compute marginal likelihood
    marginal_likeli = compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)
    
    # Compute VAF (Variance Accounted For)
    vaf = compute_vaf(y_test_noisy, y_sample_mu, doprint=True)
    
    # Compute RMSE
    rmse = compute_rmse(y_test_noisy, y_sample_mu, doprint=True)
    
    print("\n--- Computing Advanced Metrics ---")
    
    # 1. KL Divergence per dimension
    kl_div = compute_kl_divergence_per_dim(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)
    
    # 2. Confidence-adjusted RMSE per dimension
    conf_adj_rmse = compute_confidence_adjusted_error_per_dim(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)
    
    # 3. Log-likelihood per dimension
    log_likelihood = compute_log_likelihood_per_dim(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)
    
    # Create visualizations for the new metrics
    plot_dimension_metrics(kl_div, conf_adj_rmse, log_likelihood, options, path_general, file_name_general)
    plot_calibration_curves(y_test_noisy, y_sample_mu, y_sample_sigma, options, path_general, file_name_general)
  
    # Log test metrics to visualization backends
    if test_rank == 0:
        test_metrics = {
            'marginal_likelihood': marginal_likeli,
            'vaf': vaf,
            'rmse': rmse,
            'kl_divergence_avg': np.nanmean(kl_div),
            'conf_adj_rmse_avg': np.nanmean(conf_adj_rmse),
            'log_likelihood_avg': np.nanmean(log_likelihood)
        }
        log_to_visualization(writers, test_metrics, prefix='test')
        
        # Log per-dimension metrics
        for i in range(len(kl_div)):
            if not np.isnan(kl_div[i]):
                dimension_metrics = {
                    f"dimension_{i+1}/kl_divergence": kl_div[i],
                    f"dimension_{i+1}/conf_adj_rmse": conf_adj_rmse[i],
                    f"dimension_{i+1}/log_likelihood": log_likelihood[i]
                }
                log_to_visualization(writers, dimension_metrics)
        
        # Create interactive wandb visualizations if enabled
        if 'wandb' in writers and writers['wandb']:
            # Log dimension metrics table
            log_metrics_by_dimension_to_wandb(kl_div, conf_adj_rmse, log_likelihood)
            
            # Extract and log calibration curve data
            dimensions = y_test.shape[1]
            valid_dims = [i for i in range(dimensions) if not np.isnan(kl_div[i])]
            
            if valid_dims:
                all_confidence_levels = []
                all_actual_coverage = []
                
                for i in valid_dims:
                    conf_levels, actual_cov, _ = compute_calibration_curve(y_test_noisy, y_sample_mu, y_sample_sigma, i)
                    all_confidence_levels.append(conf_levels)
                    all_actual_coverage.append(actual_cov)
                
                # Use the first dimension's confidence levels (they're all the same)
                log_calibration_table_to_wandb(all_confidence_levels[0], all_actual_coverage, len(valid_dims), valid_dims)
            
            # Log figures to wandb
            figure_paths = {
                "visualizations/time_sequence": f"{path_general}time_sequences/{file_name_general}_timesequence.png",
                "visualizations/dimension_metrics": f"{path_general}dimension_metrics/{file_name_general}_dimension_metrics.png",
                "visualizations/calibration_curves": f"{path_general}calibration/{file_name_general}_calibration_curves.png",
                "visualizations/loss_curve": f"{path_general}{file_name_general}_losscurve.png"
            }
            
            for name, fig_path in figure_paths.items():
                if os.path.exists(fig_path):
                    wandb.log({name: wandb.Image(fig_path)})

    # Collect data for dataframe
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
                'rmse': rmse,
                'kl_div_avg': np.nanmean(kl_div),
                'conf_adj_rmse_avg': np.nanmean(conf_adj_rmse),
                'log_likelihood_avg': np.nanmean(log_likelihood)
                }
    # dataframe
    df.update(options_dict)
    df.update(test_dict)
    
    # Finish wandb run if enabled
    if test_rank == 0 and 'wandb' in writers and writers['wandb']:
        wandb.finish()

    return df


def main(arg):
    print("Starting...")
    args = parser.parse_args(arg)

    ngpus_per_node = torch.cuda.device_count()
    os.environ["NCCL_DEBUG"] = "INFO"
    """
    This next line is the key to getting DistributedDataParallel working on SLURM:
    SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the
    current process inside a node and is also 0 or 1 in this example.
    """

    print("echo GPUs per node: {}".format(torch.cuda.device_count()))
    print("local ID: ", os.environ.get("SLURM_LOCALID"), " node ID: ", os.environ.get("SLURM_NODEID"),
          "number of tasks: ", os.environ.get("SLURM_NTASKS"))
    local_rank = int(os.environ.get("SLURM_LOCALID"))

    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    print('cuda visible: ', os.environ.get('CUDA_VISIBLE_DEVICES'))
    proc_id = int(os.environ.get("SLURM_PROCID"))
    job_id = int(os.environ.get("SLURM_JOBID"))
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES"))
    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))
    current_device = local_rank

    torch.cuda.set_device(current_device)
    device = torch.device('cuda', local_rank if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    """
    this block initializes a process group and initiate communications
    between all processes running on all nodes
    """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    print(f"init_method = {args.init_method}")

    # local test use, remove this when running on server
    # rank = 0
    # current_device = 0
    # device = torch.device('cuda', 0 if torch.cuda.is_available() else 'cpu')

    # init the process group
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=rank,
                            timeout=timedelta(0, 240)  # 240s connection timeout
                            )

    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
    print("echo final check; ngpus_per_node={},local_rank={},rank={},available_gpus={},current_device={}"
          .format(ngpus_per_node, local_rank, rank, available_gpus, current_device))

    # Get saving path
    input_filename = "transition_data/gym_transition_inputs.pt"
    target_filename = "transition_data/gym_transition_outputs.pt"
    #parentfolder = os.path.dirname(os.getcwd())
    parentfolder = os.getcwd()
    
    # Read input data
    variable_episodes = torch.load(os.path.join(parentfolder, input_filename))
    final_next_state = torch.load(os.path.join(parentfolder, target_filename))

    print(f" shape of input data state and action: {variable_episodes.shape}")
    print(f" shape of output data next state and reward: {final_next_state.shape}")
    path_general = parentfolder + '/run/'
    if not os.path.exists(path_general):
        os.makedirs(path_general)
    else:
        pass
    file_name_general = 'Mujoco'

    # Get saving file names
    file_general_path = parentfolder + '/run/'
    if not os.path.exists(file_general_path):
        os.makedirs(file_general_path)
    else:
        pass
    
    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    plt.ion()
    train_ratio = 0.20
    validation_ratio = 0.15
    test_ratio = 0.65
    
    # Validate input data
    assert torch.isfinite(variable_episodes).any()
    assert torch.isfinite(final_next_state).any()
    assert not torch.isnan(final_next_state).any()
    assert not torch.isnan(variable_episodes).any()
    
    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(variable_episodes.permute(0, 2, 1),
                                                        final_next_state.permute(0, 2, 1), test_size=1 - train_ratio)

    # Validation/test split
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    print(f" size of training input {x_train.shape}")
    print(f" size of test input {x_test.shape}, validation data size {x_val.shape}")
    print(f" training output {y_train.shape}, test output {y_test.shape}")

    train_x, train_y, test_x, test_y, validation_x, validation_y = x_train, y_train, x_test, y_test, x_val, y_val

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
    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              worker_init_fn=seed_worker,
                              generator=g)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(validation_x, validation_y)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    normalizer_input, normalizer_output = compute_normalizer(train_x.transpose(1, 2).cpu(),
                                                            train_y.transpose(1, 2).cpu())

    # Override model configuration from command line if provided
    h_dim = args.h_dim
    z_dim = args.z_dim
    n_layers = args.n_layers
    n_mixtures = args.n_mixtures
    
    # Define model
    modelstate = ModelState(seed=seed,
                          nu=u_dim,
                          ny=y_dim,
                          h_dim=h_dim,
                          z_dim=z_dim,
                          n_layers=n_layers,
                          n_mixtures=n_mixtures,
                          sequence_length=seq_len,
                          normalizer_input=normalizer_input,
                          normalizer_output=normalizer_output
                          )
    modelstate.model.cuda()
    
    # Setup distributed data parallel
    modelstate.model = torch.nn.parallel.DistributedDataParallel(modelstate.model, find_unused_parameters=True,
                                                              device_ids=[current_device])
    print('passed distributed data parallel call')
    
    # Create an empty dataframe to store metrics
    df = {}  # allocation
    
    # Create filename based on model configuration
    file_name_general = file_name_general + '_VRNN_h{}_z{}_n{}'.format(modelstate.h_dim, modelstate.z_dim,
                                                                    modelstate.n_layers)

    # Initialize visualization backends
    model_config = {
        'h_dim': modelstate.h_dim,
        'z_dim': modelstate.z_dim,
        'n_layers': modelstate.n_layers,
        'n_mixtures': modelstate.n_mixtures,
        'sequence_length': seq_len,
        'path_general': path_general
    }
    writers = init_visualization(args, rank, model_config)

    # Train the model
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
                  batch_size=args.batch_size,
                  writers=writers
                  )

    # Test the model
    df = run_test(seed=seed,
                nu=u_dim,
                ny=y_dim,
                seq_len=seq_len,
                loaders=test_loader,
                df=df,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                path_general=path_general,
                file_name_general=file_name_general,
                batch_size=args.batch_size,
                test_rank=rank,
                writers=writers)

    # Save results to CSV
    df = pd.DataFrame(df)
    file_name = file_general_path + '_VRNN_GMM_GYM_TEST.csv'
    df.to_csv(file_name)
    
    print(f"Training and evaluation complete. Results saved to {file_name}")


if __name__ == "__main__":
    main(sys.argv[1:])