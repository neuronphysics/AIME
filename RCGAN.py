from tqdm import tqdm
import os
import numpy as np
from torchgan.models import Generator, Discriminator
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.optim as optim
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statistics import mean
from math import floor
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from VRNN.Blocks import init_weights


###########################
# based on:https://github.com/chris-hzc/Conditional_Sig_Wasserstein_GANs/blob/d022cd6dcc2ea948c3cf35c585b29c198f539754/Conditional_Sig_Wasserstein_GANs_master/lib/plot.py
def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.
    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def compare_cross_corr(x_real, x_fake):
    """ Computes cross correlation matrices of x_real and x_fake and plots them. """
    x_real = x_real.reshape(-1, x_real.shape[2])
    x_fake = x_fake.reshape(-1, x_fake.shape[2])
    cc_real = np.corrcoef(to_numpy(x_real).T)
    cc_fake = np.corrcoef(to_numpy(x_fake).T)

    vmin = min(cc_fake.min(), cc_real.min())
    vmax = max(cc_fake.max(), cc_real.max())

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cc_real, vmin=vmin, vmax=vmax)
    im = axes[1].matshow(cc_fake, vmin=vmin, vmax=vmax)

    axes[0].set_title('Real')
    axes[1].set_title('Generated')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    bin_edges = ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=bin_edges, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-pdf')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('pdf')
    return ax


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).detach().cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).detach().cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i], lb[:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


def plot_summary(x_fake, x_real, max_lag=None, labels=None):
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 1], log=True)
        compare_acf(x_real=x_real_i, x_fake=x_fake_i, ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))


def tile(t, length):
    ''' Creates an extra dimension on the tensor t and
    repeats it throughout.'''
    return t.view(-1, 1).repeat(1, length)


def calc_conv_output_length(conv_layer,
                            input_length):
    def _maybe_slice(x):
        return x[0] if isinstance(x, tuple) else x

    l = input_length
    p = _maybe_slice(conv_layer.padding)
    d = _maybe_slice(conv_layer.dilation)
    k = _maybe_slice(conv_layer.kernel_size)
    s = _maybe_slice(conv_layer.stride)
    return floor((l + 2 * p - d * (k - 1) - 1) / s + 1)


class RGANGenerator(Generator):
    def __init__(self,
                 sequence_length,
                 output_size,
                 hidden_size=None,
                 noise_size=None,
                 num_layers=1,
                 dropout=0,
                 input_size=None,
                 last_layer=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        """Recursive GAN (Generator) implementation with RNN cells.
        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, noise_size)
                output: (batch_size, sequence_length, hidden_size)
            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, output_size)
            last_layer (optional)
                input: (batch_size, sequence_length, output_size)
        Args:
            sequence_length (int): Number of points in the sequence.
                Defined by the real wm_image_replay_buffer.
            output_size (int): Size of output (usually the last tensor dimension).
                Defined by the real wm_image_replay_buffer.
            hidden_size (int, optional): Size of RNN output.
                Defaults to output_size.
            noise_size (int, optional): Size of noise used to generate fake data.
                Defaults to output_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            dropout (float, optional): Dropout probability for rnn layers.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
            rnn_type (str, optional): Type of RNN layer. Valid values are 'lstm',
                'gru' and 'rnn', the latter being the default.
            input_size (int, optional): Input size of RNN, defaults to noise_size.
            last_layer (Module, optional): Last layer of the discriminator.
        """

        # Defaults
        noise_size = noise_size or output_size
        input_size = input_size or noise_size
        hidden_size = hidden_size or output_size
        self.device = device
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_size = noise_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.label_type = "none"

        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Total size of z that will be sampled. Later, in the forward
        # method, we resize to (batch_size, sequence_length, noise_size).
        # TODO: Any resizing of z is valid as long as the total size
        #       remains sequence_length*noise_size. How does this affect
        #       the performance of the RNN?
        self.encoding_dims = sequence_length * noise_size

        super(RGANGenerator, self).__init__(self.encoding_dims,
                                            self.label_type)

        # Build RNN layer

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        # self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.last_layer = last_layer

        # Initialize all weights.
        # nn.init.xavier_normal_(self.rnn)
        self.rnn.apply(init_weights)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, z, reshape=True):
        if reshape:
            z = z.view(-1, self.sequence_length, self.noise_size)
        h0 = torch.randn((self.num_layers, z.size(0), self.hidden_size)).to(self.device)
        c0 = torch.randn((self.num_layers, z.size(0), self.hidden_size)).to(self.device)
        length = torch.LongTensor([torch.max((z[i, :, 0] != 0).nonzero()).item() + 1 for i in range(z.shape[0])])
        packed = nn.utils.rnn.pack_padded_sequence(
            z, length, batch_first=True, enforce_sorted=False
        )
        out_packed, (_, _) = self.rnn(packed, (h0, c0))
        y, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        y = self.dropout(y)
        # y = self.batchnorm(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.linear(y)
        return y if self.last_layer is None else self.last_layer(y)


class RGANDiscriminator(Discriminator):
    def __init__(self,
                 sequence_length,
                 input_size,
                 hidden_size=None,
                 num_layers=1,
                 dropout=0,
                 last_layer=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        """Recursive GAN (Discriminator) implementation with RNN cells.
        Layers:
            RNN (with activation, multiple layers):
                input:  (batch_size, sequence_length, input_size)
                output: (batch_size, sequence_length, hidden_size)
            Linear (no activation, weights shared between time steps):
                input:  (batch_size, sequence_length, hidden_size)
                output: (batch_size, sequence_length, 1)
            last_layer (optional)
                input: (batch_size, sequence_length, 1)
        Args:
            sequence_length (int): Number of points in the sequence.
            input_size (int): Size of input (usually the last tensor dimension).
            hidden_size (int, optional): Size of hidden layers in rnn.
                If None, defaults to input_size.
            num_layers (int, optional): Number of stacked RNNs in rnn.
            dropout (float, optional): Dropout probability for rnn layers.
            rnn_nonlinearity (str, optional): Non-linearity of the RNN. Must be
                either 'tanh' or 'relu'. Only valid if rnn_type == 'rnn'.
            rnn_type (str, optional): Type of RNN layer. Valid values are 'lstm',
                'gru' and 'rnn', the latter being the default.
            last_layer (Module, optional): Last layer of the discriminator.
        """

        # TODO: Insert non-linearities between Linear layers.
        # TODO: Add BatchNorm and Dropout as in https://arxiv.org/abs/1905.05928v1

        # Set hidden_size to input_size if not specified
        hidden_size = hidden_size or input_size
        self.device = device
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.label_type = "none"
        # Set kwargs (might overried above attributes)
        for key, value in kwargs.items():
            setattr(self, key, value)

        super(RGANDiscriminator, self).__init__(self.input_size,
                                                self.label_type)

        # Build RNN layer

        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.last_layer = last_layer

        # Initialize all weights.
        self.rnn.apply(init_weights)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x, length=None):
        h0 = torch.randn((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c0 = torch.randn((self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        if length is None:
            length = torch.LongTensor([torch.max((x[i, :, 0] != 0).nonzero()).item() + 1 for i in range(x.shape[0])])

        packed = nn.utils.rnn.pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        out_packed, (_, _) = self.rnn(packed, (h0, c0))
        y, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        y = self.dropout(y)
        y = self.linear(y)
        return y if self.last_layer is None else self.last_layer(y)


def gen_noise(size):
    return torch.randn(size=size)


def add_handler(logger, handlers):
    for handler in handlers:
        logger.addHandler(handler)


def setup_logging(time_logging_file, config_logging_file):
    # SET UP LOGGING
    config_logger = logging.getLogger("config_logger")
    config_logger.setLevel(logging.INFO)
    # config_logger.setLevel(logging.INFO)
    time_logger = logging.getLogger("time_logger")
    time_logger.setLevel(logging.INFO)
    # time_logger.setLevel(logging.INFO)
    # set up time handler
    time_formatter = logging.Formatter('%(asctime)s:%(message)s')
    time_handler = logging.FileHandler(time_logging_file)
    time_handler.setLevel(logging.INFO)
    time_handler.setFormatter(time_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(time_formatter)
    add_handler(time_logger, [time_handler, stream_handler])
    # setup config handler
    config_formatter = logging.Formatter('%(message)s')
    config_handler = logging.FileHandler(config_logging_file)
    config_handler.setLevel(logging.INFO)
    config_handler.setFormatter(config_formatter)
    config_logger.addHandler(config_handler)
    return config_logger, time_logger


def calculate_mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


class RCGAN:
    """RCGAN Class
    """

    @property
    def name(self):
        return 'RCGAN'

    def __init__(self,
                 train_loader,
                 device,
                 lr=0.0005,
                 noise_dim=5,
                 batch_size=28,
                 hidden_size_gen=100,
                 num_layer_gen=1,
                 hidden_size_dis=100,
                 num_layer_dis=1,
                 beta1=0.5,
                 checkpoint_dir="",
                 time_logging_file="",
                 config_logging_file="",
                 ):
        self.config_logger, self.time_logger = setup_logging(time_logging_file, config_logging_file)
        # Initalize variables.
        self.real_train_dl = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # B, T, D
        # initiate models
        # calculate variables
        # TODO: what do we need sequence length for?
        self.sequence_length = train_loader.wm_image_replay_buffer.tensors[1].shape[1],
        self.noise_dim = noise_dim
        self.attribute_size = train_loader.wm_image_replay_buffer.tensors[0].shape[-1]
        num_features = train_loader.wm_image_replay_buffer.tensors[1].shape[-1]

        self.generator = RGANGenerator(sequence_length=self.sequence_length[0],
                                       output_size=num_features,
                                       hidden_size=hidden_size_gen,
                                       noise_size=self.noise_dim + self.attribute_size,
                                       num_layers=num_layer_gen
                                       )
        self.discriminator = RGANDiscriminator(sequence_length=self.sequence_length[0],
                                               input_size=self.attribute_size + num_features,
                                               hidden_size=hidden_size_dis,
                                               num_layers=num_layer_dis
                                               )
        self.config_logger.info("DISCRIMINATOR: {0}".format(self.discriminator))
        self.generator = self.generator.to(self.device)
        self.config_logger.info("GENERATOR: {0}".format(self.generator))
        self.discriminator = self.discriminator.to(self.device)
        # loss

        # Setup optimizer
        self.optimizer_dis = optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("DISCRIMINATOR OPTIMIZER: {0}".format(self.optimizer_dis))
        self.optimizer_gen = optim.AdamW(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.config_logger.info("GENERATOR OPTIMIZER: {0}".format(self.optimizer_gen))
        self.batch_size = batch_size
        self.config_logger.info("Batch Size: {0}".format(self.batch_size))
        self.config_logger.info("Noise Dimension: {0}".format(self.noise_dim))
        self.config_logger.info("d_rounds: {0}".format("1"))
        self.config_logger.info("g_rounds: {0}".format("1"))
        self.config_logger.info("Device: {0}".format(self.device))
        self.config_logger.info("Input Dimension: {0}".format(self.attribute_size))
        self.config_logger.info("Output Dimension: {0}".format(num_features))
        self.config_logger.info("Sequence Length: {0}".format(self.sequence_length))

    def wgan_gp_reg(self, x_real, x_fake, center=1., lambda_gp=10.0):
        # eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        batch_size = x_real.shape[0]
        eps = torch.rand(batch_size, 1, 1).to(self.device)
        eps = eps.expand_as(x_real)
        # eps = torch.randn_like(x_real).to(self.device)
        x_interp = eps * x_real + (1 - eps) * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp)

        gradients = torch.autograd.grad(inputs=x_interp,
                                        outputs=d_out,
                                        grad_outputs=torch.ones_like(d_out, requires_grad=False, device=self.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        )[0]

        gradients = gradients.view(gradients.size(), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - center) ** 2).mean() * lambda_gp
        return gradient_penalty

    def save(self, epoch):
        if not os.path.exists("{0}/epoch_{1}".format(self.checkpoint_dir, epoch)):
            os.makedirs("{0}/epoch_{1}".format(self.checkpoint_dir, epoch))
        torch.save(self.generator, "{0}/epoch_{1}/generator.pth".format(self.checkpoint_dir, epoch))
        torch.save(self.discriminator, "{0}/epoch_{1}/discriminator.pth".format(self.checkpoint_dir, epoch))

    def load(self, model_dir=None):
        if not os.path.exists(model_dir):
            raise Exception("Directory to load pytorch model doesn't exist")
        self.generator = torch.load("{0}/generator.pth".format(model_dir))
        self.discriminator = torch.load("{0}/discriminator.pth".format(model_dir))
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

    def inference(self, epoch, model_dir=None):
        if model_dir is None:
            model_dir = "{0}/epoch_{1}".format(self.checkpoint_dir, epoch)
        batch_size = self.batch_size

        while self.real_train_dl.wm_image_replay_buffer.tensors[0].shape[0] % batch_size != 0:
            batch_size -= 1
        rounds = self.real_train_dl.wm_image_replay_buffer.tensors[0].shape[0] // batch_size
        sampled_features = np.zeros(
            (0, self.real_train_dl.wm_image_replay_buffer.tensors[1].shape[1],
             self.real_train_dl.wm_image_replay_buffer.tensors[1].shape[-1] - 2))
        sampled_attributes = np.zeros(
            (0, self.real_train_dl.wm_image_replay_buffer.tensors[1].shape[1],
             self.real_train_dl.wm_image_replay_buffer.tensors[0].shape[-1]))
        sampled_gen_flags = np.zeros((0, self.real_train_dl.wm_image_replay_buffer.tensors[1].shape[1]))
        sampled_lengths = np.zeros(0)
        for i in range(rounds):
            features, attributes, gen_flags, lengths = self.sample_from(batch_size=batch_size)
            target_length = self.real_train_dl.wm_image_replay_buffer.tensors[1].shape[1]
            features = pad_along_axis(features, target_length, axis=1)
            gen_flags = pad_along_axis(gen_flags, target_length, axis=1)
            sampled_features = np.concatenate((sampled_features, features), axis=0)
            sampled_attributes = np.concatenate((sampled_attributes, attributes), axis=0)
            sampled_gen_flags = np.concatenate((sampled_gen_flags, gen_flags), axis=0)
            sampled_lengths = np.concatenate((sampled_lengths, lengths), axis=0)
        np.savez("{0}/generated_samples.npz".format(model_dir), sampled_features=sampled_features,
                 sampled_attributes=sampled_attributes, sampled_gen_flags=sampled_gen_flags,
                 sampled_lengths=sampled_lengths)

    def sample_from(self, batch_size, return_gen_flag_feature=False):
        self.discriminator.eval()
        self.generator.eval()
        noise = gen_noise((batch_size, self.sequence_length[0], self.noise_dim)).to(self.device)
        attributes, data_feature = next(iter(self.real_train_dl))
        attributes = attributes.to(self.device)
        attributes = attributes[:batch_size, :]
        noise = torch.cat((attributes, noise), dim=2)

        with torch.no_grad():
            features = self.generator(noise)
            features = features.cpu().numpy()
            gen_flags = np.zeros(features.shape[:-1])
            lengths = np.zeros(features.shape[0])
            for i in range(len(features)):
                winner = (features[i, :, -1] > features[i, :, -2])
                argmax = np.argmax(winner == True)
                if argmax == 0:
                    gen_flags[i, :] = 1
                else:
                    gen_flags[i, :argmax + 1] = 1
                lengths[i] = argmax
            if not return_gen_flag_feature:
                features = features[:, :, :-2]
        return features, attributes.cpu().numpy(), gen_flags, lengths

    def train(self, epochs, writer_frequency=1, saver_frequency=20):
        avg_mmd = []
        epoch_loss = 0
        total_samples = 0
        tqdm_train_descr_format = "Training RCGAN model: Epoch  Loss = {:.8f}"
        tqdm_train_descr = tqdm_train_descr_format.format(0, float('inf'))
        tqdm_train_obj = tqdm(range(epochs), desc=tqdm_train_descr)
        for epoch in tqdm_train_obj:
            mmd = []
            for batch_idx, (data_attribute, data_feature) in enumerate(self.real_train_dl):
                data_attribute = data_attribute.to(self.device)
                input_feature = data_feature.to(self.device)
                out_real = input_feature  ###
                batch_size = data_attribute.shape[0]
                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                noise = gen_noise((batch_size, self.sequence_length[0], self.noise_dim)).to(self.device)
                noise = torch.cat((data_attribute, noise), dim=2)
                input_feature = torch.cat((data_attribute, input_feature), dim=2)
                fake = self.generator(noise)

                out_fake = fake.clone()  ###

                x = out_fake.permute(0, 2, 1)
                padded = nn.ConstantPad1d((0, input_feature.shape[1] - fake.shape[1]), 0)(x)
                x = padded.permute(0, 2, 1)
                mmd.append(calculate_mmd_rbf(torch.mean(x, dim=0).detach().cpu().numpy(),
                                             torch.mean(data_feature, dim=0).detach().cpu().numpy()))
                fake = torch.cat((data_attribute, x), dim=2)

                length_real = torch.LongTensor([torch.max((input_feature[i, :, 0] != 0).nonzero()).item() + 1 for i in
                                                range(input_feature.shape[0])])
                max_seq = torch.max(length_real)

                output_real = torch.cat((data_attribute, data_feature), dim=2)

                disc_real = self.discriminator(input_feature, length_real)

                disc_fake = self.discriminator(fake.to(self.device))
                gradient_penalty = self.wgan_gp_reg(output_real.to(self.device), fake)
                d_loss = -torch.mean(disc_real) + torch.mean(disc_fake) + gradient_penalty
                self.discriminator.zero_grad()
                self.generator.zero_grad()
                d_loss.backward(retain_graph=True)

                self.optimizer_dis.step()
                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                self.optimizer_gen.step()
                total_samples += batch_size
                epoch_loss += d_loss.item()
            epoch_loss = epoch_loss / total_samples
            tqdm_descr = tqdm_train_descr_format.format(epoch_loss)
            tqdm_train_obj.set_description(tqdm_descr)
            avg_mmd.append(mean(mmd))
            self.time_logger.info('END OF EPOCH {0}'.format(epoch))
            if epoch % saver_frequency == 0:
                self.save(epoch)
                self.inference(epoch)
                self.discriminator.train()
                self.generator.train()
                length = torch.LongTensor(
                    [torch.max((output_real[i, :, 0] != 0).nonzero()).item() + 1 for i in range(output_real.shape[0])])

                plot_summary(out_fake, out_real, max_lag=None)
                plt.savefig('{}/summary_features_epoch_{}_RCGAN.png'.format(self.checkpoint_dir, epoch))
        np.save("{}/mmd.npy".format(self.checkpoint_dir), np.asarray(avg_mmd))


if __name__ == "__main__":
    # get saving path
    # get saving file names
    gan_type = "RCGAN"
    checkpoint_dir = os.getcwd() + '/sac/data/'
    time_logging_file = '{}/time.log'.format(checkpoint_dir)
    config_logging_file = '{}/config.log'.format(checkpoint_dir)
    checkpoint_dir = '{}/checkpoint'.format(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        pass
    seed = 1234
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_ratio = 0.34
    validation_ratio = 0.33
    test_ratio = 0.33

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(variable_episodes, final_next_state, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio))
    # scale the wm_image_replay_buffer

    print(f" size of training input {x_train.shape}, test input {x_test.shape}, validation data size {x_val.shape}")
    print(f" training output {y_train.shape}, test output {y_test.shape}")
    if torch.cuda.is_available():
        x_train, y_train, x_test, y_test, x_val, y_val = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda(), x_val.cuda(), y_val.cuda()

    save_frequency = 50
    batch_size = 50
    train_dataset = TensorDataset(x_train, y_train)
    # we don't shuffle because it is a time series data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    epoch = 501

    valid_dataset = TensorDataset(x_val, y_val)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    trainer = RCGAN(train_loader, device=device, checkpoint_dir=checkpoint_dir,
                    time_logging_file=time_logging_file, batch_size=batch_size,
                    config_logging_file=config_logging_file)

    trainer.train(epochs=epoch, writer_frequency=1, saver_frequency=save_frequency)
