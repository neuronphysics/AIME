import numpy as np
import torch
from torch.distributions import Normal
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KernelDensity

def compute_kl_divergence_per_dim(y, yhat_mu, yhat_sigma, kde_bandwidth=0.1, n_samples=1000, doprint=False):
    """
    Computes the KL divergence between predicted distributions and empirical data distributions
    for each output dimension.
    
    Args:
        y: Ground truth values, shape (batch_size, output_dim, seq_len)
        yhat_mu: Predicted means, shape (batch_size, output_dim, seq_len)
        yhat_sigma: Predicted standard deviations, shape (batch_size, output_dim, seq_len)
        kde_bandwidth: Bandwidth for kernel density estimation of true distribution
        n_samples: Number of samples to use for KL estimation
        doprint: Whether to print results
        
    Returns:
        kl_div: KL divergence for each dimension, shape (output_dim,)
    """
    num_outputs = y.shape[1]
    kl_div = np.zeros([num_outputs])
    
    # Compute valid sequence lengths for each batch item
    seq_len = [np.max(np.where(~np.isnan(y[i, 0, :]))[0]) + 1 if np.any(~np.isnan(y[i, 0, :])) else 0 
               for i in range(y.shape[0])]
    
    for i in range(num_outputs):
        # Collect all valid ground truth values for this dimension
        y_valid = []
        for b in range(y.shape[0]):
            for t in range(seq_len[b]):
                if not np.isnan(y[b, i, t]):
                    y_valid.append(y[b, i, t])
        
        if len(y_valid) == 0:
            kl_div[i] = np.nan
            continue
            
        y_valid = np.array(y_valid).reshape(-1, 1)
        
        # Fit kernel density estimate to empirical data
        kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
        kde.fit(y_valid)
        
        # Monte Carlo estimation of KL divergence
        kl_sum = 0
        count = 0
        
        for b in range(y.shape[0]):
            for t in range(seq_len[b]):
                if not np.isnan(y[b, i, t]) and yhat_sigma[b, i, t] > 0:
                    # Generate samples from predicted distribution
                    pred_dist = stats.norm(yhat_mu[b, i, t], yhat_sigma[b, i, t])
                    samples = pred_dist.rvs(n_samples).reshape(-1, 1)
                    
                    # Compute log densities
                    log_pred = pred_dist.logpdf(samples.flatten())
                    log_true = kde.score_samples(samples)  # Returns log density
                    
                    # KL = E_p[log(p/q)] = E_p[log p - log q]
                    kl = np.mean(log_pred - log_true)
                    kl_sum += kl
                    count += 1
        
        if count > 0:
            kl_div[i] = kl_sum / count
        else:
            kl_div[i] = np.nan
    
    if doprint:
        print("\nKL Divergence Per Dimension:")
        for i in range(num_outputs):
            if not np.isnan(kl_div[i]):
                print(f'  Dim {i+1}: {kl_div[i]:.4f}')
            else:
                print(f'  Dim {i+1}: N/A (insufficient data)')
    
    return kl_div

def compute_confidence_adjusted_error_per_dim(y, yhat_mu, yhat_sigma, doprint=False):
    """
    Compute errors weighted by the predicted confidence (inverse variance) for each dimension.
    Lower confidence predictions (higher sigma) contribute less to the error metric.
    
    Args:
        y: Ground truth values, shape (batch_size, output_dim, seq_len)
        yhat_mu: Predicted means, shape (batch_size, output_dim, seq_len)
        yhat_sigma: Predicted standard deviations, shape (batch_size, output_dim, seq_len)
        doprint: Whether to print results
        
    Returns:
        conf_adj_rmse: Confidence-adjusted RMSE for each dimension
    """
    num_outputs = y.shape[1]
    conf_adj_mse = np.zeros([num_outputs])
    
    # Compute valid sequence lengths for each batch item
    seq_len = [np.max(np.where(~np.isnan(y[i, 0, :]))[0]) + 1 if np.any(~np.isnan(y[i, 0, :])) else 0 
               for i in range(y.shape[0])]
    
    for i in range(num_outputs):
        total_weighted_sq_error = 0.0
        total_weight = 0.0
        
        for b in range(y.shape[0]):
            for t in range(seq_len[b]):
                if not np.isnan(y[b, i, t]) and yhat_sigma[b, i, t] > 0:
                    # Squared error
                    sq_error = (y[b, i, t] - yhat_mu[b, i, t]) ** 2
                    
                    # Weight by precision (inverse variance)
                    precision = 1.0 / (yhat_sigma[b, i, t] ** 2 + 1e-10)
                    
                    total_weighted_sq_error += sq_error * precision
                    total_weight += precision
        
        if total_weight > 0:
            conf_adj_mse[i] = total_weighted_sq_error / total_weight
        else:
            conf_adj_mse[i] = np.nan
    
    conf_adj_rmse = np.sqrt(conf_adj_mse)
    
    if doprint:
        print("\nConfidence-Adjusted RMSE Per Dimension:")
        for i in range(num_outputs):
            if not np.isnan(conf_adj_rmse[i]):
                print(f'  Dim {i+1}: {conf_adj_rmse[i]:.4f}')
            else:
                print(f'  Dim {i+1}: N/A (insufficient data)')
    
    return conf_adj_rmse

def compute_log_likelihood_per_dim(y, yhat_mu, yhat_sigma, doprint=False):
    """
    Compute log-likelihood of true values under predicted distributions for each dimension.
    
    Args:
        y: Ground truth values, shape (batch_size, output_dim, seq_len)
        yhat_mu: Predicted means, shape (batch_size, output_dim, seq_len)
        yhat_sigma: Predicted standard deviations, shape (batch_size, output_dim, seq_len)
        doprint: Whether to print results
        
    Returns:
        log_likelihood: Log-likelihood for each dimension
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(y, torch.Tensor):
        y_tensor = torch.tensor(y, dtype=torch.float64)
        mu_tensor = torch.tensor(yhat_mu, dtype=torch.float64)
        sigma_tensor = torch.tensor(yhat_sigma, dtype=torch.float64)
    else:
        y_tensor = y
        mu_tensor = yhat_mu
        sigma_tensor = yhat_sigma
    
    num_outputs = y.shape[1]
    log_likelihood = np.zeros([num_outputs])
    
    # Compute valid sequence lengths for each batch item
    seq_len = [torch.max((~torch.isnan(y_tensor[i, 0, :])).nonzero()).item() + 1 
               if torch.any(~torch.isnan(y_tensor[i, 0, :])) else 0 
               for i in range(y_tensor.shape[0])]
    
    for i in range(num_outputs):
        total_log_prob = 0.0
        count = 0
        
        for b in range(y_tensor.shape[0]):
            for t in range(seq_len[b]):
                if not torch.isnan(y_tensor[b, i, t]) and sigma_tensor[b, i, t] > 0:
                    # Create distribution for this prediction
                    dist = Normal(mu_tensor[b, i, t], sigma_tensor[b, i, t])
                    
                    # Compute log probability
                    log_prob = dist.log_prob(y_tensor[b, i, t])
                    
                    if not torch.isnan(log_prob):
                        total_log_prob += log_prob.item()
                        count += 1
        
        if count > 0:
            log_likelihood[i] = total_log_prob / count
        else:
            log_likelihood[i] = np.nan
    
    if doprint:
        print("\nAverage Log-Likelihood Per Dimension:")
        for i in range(num_outputs):
            if not np.isnan(log_likelihood[i]):
                print(f'  Dim {i+1}: {log_likelihood[i]:.4f}')
            else:
                print(f'  Dim {i+1}: N/A (insufficient data)')
    
    return log_likelihood

def plot_dimension_metrics(kl_div, conf_adj_rmse, log_likelihood, options, path_general, file_name_general):
    """
    Create a visualization of the per-dimension metrics.
    
    Args:
        kl_div: KL divergence for each dimension
        conf_adj_rmse: Confidence-adjusted RMSE for each dimension
        log_likelihood: Log-likelihood for each dimension
        options: Dictionary of model options/configuration
        path_general: Base path for saving figures
        file_name_general: Base file name for saving figures
    """
    # Create directory if it doesn't exist
    path = path_general + 'dimension_metrics/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    num_dims = len(kl_div)
    dim_indices = np.arange(1, num_dims + 1)
    
    # Create a figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot KL divergence
    valid_kl = ~np.isnan(kl_div)
    if np.any(valid_kl):
        axes[0].bar(dim_indices[valid_kl], kl_div[valid_kl], color='skyblue')
        axes[0].set_title('KL Divergence per Dimension', fontsize=14)
        axes[0].set_xlabel('Dimension', fontsize=12)
        axes[0].set_ylabel('KL Divergence', fontsize=12)
        axes[0].set_xticks(dim_indices)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        axes[0].text(0.5, 0.5, 'No valid KL divergence data', 
                    horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    
    # Plot Confidence-Adjusted RMSE
    valid_rmse = ~np.isnan(conf_adj_rmse)
    if np.any(valid_rmse):
        axes[1].bar(dim_indices[valid_rmse], conf_adj_rmse[valid_rmse], color='lightgreen')
        axes[1].set_title('Confidence-Adjusted RMSE per Dimension', fontsize=14)
        axes[1].set_xlabel('Dimension', fontsize=12)
        axes[1].set_ylabel('Conf-Adj RMSE', fontsize=12)
        axes[1].set_xticks(dim_indices)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, 'No valid confidence-adjusted RMSE data', 
                    horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
    
    # Plot Log-Likelihood
    valid_ll = ~np.isnan(log_likelihood)
    if np.any(valid_ll):
        axes[2].bar(dim_indices[valid_ll], log_likelihood[valid_ll], color='salmon')
        axes[2].set_title('Log-Likelihood per Dimension', fontsize=14)
        axes[2].set_xlabel('Dimension', fontsize=12)
        axes[2].set_ylabel('Log-Likelihood', fontsize=12)
        axes[2].set_xticks(dim_indices)
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    else:
        axes[2].text(0.5, 0.5, 'No valid log-likelihood data', 
                    horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
    
    # Add a title with model configuration
    fig.suptitle(f'Dimension-wise Metrics for Model (h={options["h_dim"]}, z={options["z_dim"]}, n={options["n_layers"]})', 
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the figure
    file_name = file_name_general + '_dimension_metrics.png'
    plt.savefig(path + file_name, format='png', dpi=300)
    
    if options.get('showfig', False):
        plt.show()
    else:
        plt.close(fig)

def plot_calibration_curves(y, yhat_mu, yhat_sigma, options, path_general, file_name_general):
    """
    Plot calibration curves for each dimension showing expected vs. actual coverage of prediction intervals.
    
    Args:
        y: Ground truth values, shape (batch_size, output_dim, seq_len)
        yhat_mu: Predicted means, shape (batch_size, output_dim, seq_len)
        yhat_sigma: Predicted standard deviations, shape (batch_size, output_dim, seq_len)
        options: Dictionary of model options/configuration
        path_general: Base path for saving figures
        file_name_general: Base file name for saving figures
    """
    # Create directory if it doesn't exist
    path = path_general + 'calibration/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    num_outputs = y.shape[1]
    
    # Create subplots - one row per dimension
    fig, axes = plt.subplots(num_outputs, 1, figsize=(10, 4 * num_outputs))
    if num_outputs == 1:
        axes = [axes]  # Make sure axes is a list for consistency
    
    # Confidence levels to evaluate (from 0.1 to 0.99)
    confidence_levels = np.linspace(0.1, 0.99, 30)
    
    # Compute valid sequence lengths for each batch item
    seq_len = [np.max(np.where(~np.isnan(y[i, 0, :]))[0]) + 1 if np.any(~np.isnan(y[i, 0, :])) else 0 
               for i in range(y.shape[0])]
    
    for i in range(num_outputs):
        # Collect all normalized errors for this dimension
        normalized_errors = []
        for b in range(y.shape[0]):
            for t in range(seq_len[b]):
                if not np.isnan(y[b, i, t]) and yhat_sigma[b, i, t] > 0:
                    error = (y[b, i, t] - yhat_mu[b, i, t]) / yhat_sigma[b, i, t]
                    normalized_errors.append(error)
        
        if len(normalized_errors) == 0:
            axes[i].text(0.5, 0.5, f'No valid data for dimension {i+1}', 
                        horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            continue
        
        normalized_errors = np.array(normalized_errors)
        
        # Calculate actual coverage for each confidence level
        actual_coverage = []
        for conf_level in confidence_levels:
            # Calculate the z-score for this confidence level
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            # Calculate the fraction of errors within this z-score
            coverage = np.mean(np.abs(normalized_errors) <= z_score)
            actual_coverage.append(coverage)
        
        # Plot the calibration curve
        axes[i].plot(confidence_levels, actual_coverage, 'bo-', label='Actual Coverage')
        axes[i].plot(confidence_levels, confidence_levels, 'r--', label='Ideal Calibration')
        axes[i].set_title(f'Calibration Curve for Dimension {i+1}', fontsize=14)
        axes[i].set_xlabel('Expected Coverage', fontsize=12)
        axes[i].set_ylabel('Actual Coverage', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].legend()
        
        # Plot the histogram of normalized errors as an inset
        ax_inset = axes[i].inset_axes([0.65, 0.1, 0.3, 0.3])
        ax_inset.hist(normalized_errors, bins=30, density=True, alpha=0.7)
        ax_inset.set_xlabel('Normalized Error')
        ax_inset.set_title('Error Distribution')
        
        # Overlay standard normal PDF
        x = np.linspace(-4, 4, 100)
        ax_inset.plot(x, stats.norm.pdf(x), 'r-', alpha=0.7)
    
    # Add a title with model configuration
    fig.suptitle(f'Calibration Curves for Model (h={options["h_dim"]}, z={options["z_dim"]}, n={options["n_layers"]})', 
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the figure
    file_name = file_name_general + '_calibration_curves.png'
    plt.savefig(path + file_name, format='png', dpi=300)
    
    if options.get('showfig', False):
        plt.show()
    else:
        plt.close(fig)
