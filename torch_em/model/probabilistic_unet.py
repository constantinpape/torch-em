"""@private
"""

# This code is based on the original TensorFlow implementation: https://github.com/SimonKohl/probabilistic_unet
# The below implementation is from: https://github.com/stefanknegt/Probabilistic-Unet-Pytorch

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, kl

from torch_em.model import UNet2d
from torch_em.loss.dice import DiceLossWithLogits


def truncated_normal_(tensor, mean=0, std=1):
    """@private
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block
    convolutional layers, after each block a pooling operation is performed.
    And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(
        self,
        input_channels,
        num_filters,
        no_convs_per_block,
        initializers,
        padding=True,
        posterior=False,
        num_classes=None
    ):

        super().__init__()

        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            assert num_classes is not None
            self.input_channels += num_classes

        layers = []
        output_dim = None  # Initialize output_dim of the layers

        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(
        self,
        input_channels,
        num_filters,
        no_convs_per_block,
        latent_dim,
        initializers,
        posterior=False,
        num_classes=None
    ):

        super().__init__()

        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim

        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'

        self.encoder = Encoder(
                                self.input_channels,
                                self.num_filters,
                                self.no_convs_per_block,
                                initializers,
                                posterior=self.posterior,
                                num_classes=num_classes
                            )

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        #
        # @ Original paper's training details:
        # All weights of all models are initialized with orthogonal initialization having the gain (multiplicative
        # factor) set to 1, and the bias terms are initialized by sampling from a truncated normal with σ = 0.001

        # nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')  # from Stefan's impl.
        # nn.init.normal_(self.conv_layer.weight, std=0.001)  # suggested @issues from Stefan's impl.

        # nn.init.normal_(self.conv_layer.bias)  # from Stefan's impl.
        #

        nn.init.orthogonal_(self.conv_layer.weight, gain=1)
        nn.init.trunc_normal_(self.conv_layer.bias, std=0.001)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(
        self,
        num_filters,
        latent_dim,
        num_output_channels,
        num_classes,
        no_convs_fcomb,
        initializers,
        use_tile=True,
        device=None
    ):

        super().__init__()

        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
                                    np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
                                ).to(self.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is (batch_size x latent_dim) and feature_map is (batch_size x no_channels x H x W).
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUNet(nn.Module):
    """ This network implementation for the Probabilistic UNet of Kohl et al. (https://arxiv.org/abs/1806.05034).
    This generative segmentation heuristic uses UNet combined with a conditional variational
    autoencoder enabling to efficiently produce an unlimited number of plausible hypotheses.

    The following elements are initialized to get our desired network:
    input_channels: the number of channels in the image (1 for grayscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisting of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    beta: KL and reconstruction loss are weighted using a KL weighting factor (β)
    consensus_masking: activates consensus masking in the reconstruction loss
    rl_swap: switches the reconstruction loss to dice loss from the default (binary cross-entroy loss)

    Args:
        input_channels [int] - (default: 1)
        num_classes [int] - (default: 1)
        num_filters [list] - (default: [32, 64, 128, 192])
        latent_dim [int] - (default: 6)
        no_convs_fcomb [int] - (default: 4)
        beta [float] - (default: 10.0)
        consensus_masking [bool] - (default: False)
        rl_swap [bool] - (default: False)
        device [torch.device] - (default: None)
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        latent_dim=6,
        no_convs_fcomb=4,
        beta=10.0,
        consensus_masking=False,
        rl_swap=False,
        device=None
    ):

        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0
        self.consensus_masking = consensus_masking
        self.rl_swap = rl_swap

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.unet = UNet2d(
                            in_channels=self.input_channels,
                            out_channels=None,
                            depth=len(self.num_filters),
                            initial_features=num_filters[0]
                        ).to(self.device)

        self.prior = AxisAlignedConvGaussian(
                            self.input_channels,
                            self.num_filters,
                            self.no_convs_per_block,
                            self.latent_dim,
                            self.initializers
                        ).to(self.device)

        self.posterior = AxisAlignedConvGaussian(
                            self.input_channels,
                            self.num_filters,
                            self.no_convs_per_block,
                            self.latent_dim,
                            self.initializers,
                            posterior=True,
                            num_classes=num_classes
                        ).to(self.device)

        self.fcomb = Fcomb(
                            self.num_filters,
                            self.latent_dim,
                            self.input_channels,
                            self.num_classes,
                            self.no_convs_fcomb,
                            {'w': 'orthogonal', 'b': 'normal'},
                            use_tile=True,
                            device=self.device
                        ).to(self.device)

    def _check_shape(self, patch):
        spatial_shape = tuple(patch.shape)[2:]
        depth = len(self.num_filters)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for Probabilistic U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)

    def forward(self, patch, segm=None):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        self._check_shape(patch)

        if segm is not None:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample and combining this with UNet features
        """
        if testing is False:
            # TODO: prior distribution ? (posterior in this case!)
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, consm=None, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        consm: consensus response
        """

        if self.rl_swap:
            criterion = DiceLossWithLogits()
        else:
            criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction=None)

        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(
                        self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
                    )

        # Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean,
                                               calculate_posterior=False, z_posterior=z_posterior)

        if self.consensus_masking is True and consm is not None:
            reconstruction_loss = criterion(self.reconstruction * consm, segm * consm)
        else:
            reconstruction_loss = criterion(self.reconstruction, segm)

        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl)
