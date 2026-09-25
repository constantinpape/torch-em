"""@private
"""

# This code is based on:
#   1. The original TensorFlow implementation: https://github.com/SimonKohl/probabilistic_unet
#   2. PyTorch adaptation from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
#   3. PhiSeg's benchmarking script from https://github.com/annawundram/glaucoma-diagnosis-pipeline.

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from torch_em.model import UNet2d
from torch_em.loss.dice import DiceLoss
from torch_em.model.unet import get_norm_layer


def init_weights(m: nn.Module) -> None:
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias, std=0.001)


def init_weights_orthogonal_normal(m: nn.Module) -> None:
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias, std=0.001)


class Encoder(nn.Module):
    """Convolutional encoder for the prior and posterior networks.

    Stacks len(num_filters) blocks of no_convs_per_block Conv-Norm-ReLU layers with average
    pooling between blocks. For the posterior, the segmentation mask is concatenated with the
    image along the channel axis before encoding.

    Args:
        input_channels: Number of input image channels.
        num_filters: Number of filters for each encoder block.
        no_convs_per_block: Number of Conv-Norm-ReLU layers per block.
        posterior: If True, expects a concatenated segmentation mask as input.
        num_classes: Number of segmentation channels appended when posterior=True.
        norm: Normalization type applied after each convolution. None disables normalisation.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        posterior: bool = False,
        num_classes: Optional[int] = None,
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accommodate for the mask concatenated at the channel axis, increase input_channels.
            assert num_classes is not None
            self.input_channels += num_classes

        layers = []
        output_dim = None

        for i in range(len(self.num_filters)):
            # First block: input_channels -> num_filters[i]; subsequent: prev_output -> num_filters[i].
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3))
            if norm is not None:
                layers.append(get_norm_layer(norm, 2, output_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3))
                if norm is not None:
                    layers.append(get_norm_layer(norm, 2, output_dim))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AxisAlignedConvGaussian(nn.Module):
    """Convolutional network that outputs a diagonal-covariance Gaussian distribution.

    Encodes the input (and optionally a segmentation mask) into a global feature vector,
    then predicts mu and log-sigma for each latent dimension. Returns an Independent(Normal)
    distribution over the latent space.

    Args:
        input_channels: Number of input image channels.
        num_filters: Number of filters per encoder block.
        no_convs_per_block: Number of convolutions per encoder block.
        latent_dim: Dimensionality of the latent space.
        posterior: If True, encodes both image and segmentation (posterior q(z|x,y)).
        num_classes: Number of segmentation channels when posterior=True.
        use_onehot: If True, converts integer class labels to one-hot before concatenation.
        num_raters: Number of segmentation maps per image for the joint posterior.
        norm: Normalization type passed to the encoder. None disables normalisation.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        latent_dim: int,
        posterior: bool = False,
        num_classes: Optional[int] = None,
        use_onehot: bool = False,
        num_raters: int = 1,
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_onehot = use_onehot

        self.encoder = Encoder(
            input_channels,
            num_filters,
            no_convs_per_block,
            posterior=posterior,
            num_classes=None if num_classes is None else num_classes * num_raters,
            norm=norm,
        )

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        # Orthogonal init + truncated-normal bias per the original paper's training details.
        nn.init.orthogonal_(self.conv_layer.weight, gain=1)
        nn.init.trunc_normal_(self.conv_layer.bias, std=0.001)

    def forward(self, patch: torch.Tensor, segm: Optional[torch.Tensor] = None) -> Independent:
        # Posterior: encode segm and concatenate with image along channel axis.
        # One-hot encodes class-index labels to remove spurious ordinal relationships.
        # Centering (- 0.5) keeps inputs zero-mean in both the one-hot and raw binary cases.
        if segm is not None:
            if self.use_onehot:
                segm = F.one_hot(segm.long(), self.num_classes).movedim(-1, 2).flatten(1, 2).float() - 0.5
            else:
                segm = segm.float() - 0.5
            patch = torch.cat((patch, segm), dim=1)

        encoding = self.encoder(patch)

        # Global average pool to (B, C, 1, 1), then squeeze spatial dims to (B, C).
        encoding = encoding.mean(dim=(2, 3), keepdim=True)
        mu_log_sigma = self.conv_layer(encoding).squeeze(3).squeeze(2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """A sequence of 1x1 convolutions that combines the UNet feature map with a latent sample.

    Broadcasts z to the spatial size of the feature map, concatenates along the channel axis,
    and applies no_convs_fcomb pointwise convolutions to produce segmentation logits.

    Args:
        num_filters: Filter counts from the UNet encoder; num_filters[0] sets the fcomb width.
        latent_dim: Dimensionality of the latent sample z.
        num_output_channels: Number of output segmentation channels.
        no_convs_fcomb: Total number of 1x1 conv layers (including the final projection).
    """
    def __init__(
        self,
        num_filters: List[int],
        latent_dim: int,
        num_output_channels: int,
        no_convs_fcomb: int,
    ) -> None:
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(num_filters[0] + latent_dim, num_filters[0], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(nn.Conv2d(num_filters[0], num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Conv2d(num_filters[0], num_output_channels, kernel_size=1)

        self.layers.apply(init_weights_orthogonal_normal)
        self.last_layer.apply(init_weights_orthogonal_normal)

    def forward(self, feature_map: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Combine UNet feature map with a latent sample to produce a segmentation.

        Args:
            feature_map: UNet decoder output of shape (B, C, H, W).
            z: Latent sample of shape (B, latent_dim), broadcast to (B, latent_dim, H, W)
                before concatenation with feature_map.
        """
        H, W = feature_map.shape[2], feature_map.shape[3]
        z = z.view(z.shape[0], z.shape[1], 1, 1).expand(-1, -1, H, W)
        feature_map = torch.cat((feature_map, z), dim=1)
        return self.last_layer(self.layers(feature_map))


class ProbabilisticUNet(nn.Module):
    """Network implementation for the Probabilistic UNet of Kohl et al. (https://arxiv.org/abs/1806.05034).
    This generative segmentation heuristic uses UNet combined with a conditional variational
    autoencoder enabling to efficiently produce an unlimited number of plausible hypotheses.

    Labels have shape (B, R, H, W), where R is num_raters. Each rater provides a binary mask
    or a multiclass map of integer class IDs. The joint posterior receives all raters as input channels.
    Multiclass maps become R * C one-hot channels. Reconstruction loss averages over raters.
    Prior samples have shape (B, C, H, W).

    Args:
        input_channels: Number of channels in the image (1 for grayscale and 3 for RGB). The default is set to 1.
        output_channels: Number of channels to predict. The default is set to 1.
        num_raters: Number of annotators per image for binary or multiclass labels. The default is set to 1.
        num_filters: Number of filters per encoder level. The default is set to [32, 64, 128, 192].
        latent_dim: Dimension of the latent space. The default is set to 6.
        no_convs_per_block: Number of convolutions per block in the prior/posterior encoder. The default is set to 3.
        no_convs_fcomb: Number of convolutions in the feature combination module. The default is set to 4.
        norm: Normalization type for the prior and posterior encoder. The default is set to 'InstanceNorm'.
        beta: Weighting factor for the KL divergence term in the ELBO (loss = reconstruction + beta * KL).
            beta=1.0 is the standard VAE objective with no extra regularization. The original paper
            used beta=10.0 for the LIDC-IDRI multi-rater chest X-ray task after task-specific tuning,
            which over-regularizes the latent space for most other tasks. Set beta > 1 to encourage a
            more structured latent space at the cost of reconstruction quality, or beta < 1 to
            prioritize reconstruction. The default is set to 1.0.
        consensus_masking: Whether to apply consensus masking in the reconstruction loss. The default is set to False.
        rl_swap: Whether to use dice loss instead of BCE/CE for reconstruction. The default is set to False.
        device: Device to place the model on. The default is set to None.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_raters: int = 1,
        num_filters: List[int] = [32, 64, 128, 192],
        latent_dim: int = 6,
        no_convs_per_block: int = 3,
        no_convs_fcomb: int = 4,
        norm: Optional[str] = "InstanceNorm",
        beta: float = 1.0,
        consensus_masking: bool = False,
        rl_swap: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if output_channels < 1 or num_raters < 1:
            raise ValueError("output_channels and num_raters must be positive.")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_raters = num_raters
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = no_convs_per_block
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.consensus_masking = consensus_masking
        self.rl_swap = rl_swap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.unet = UNet2d(
            in_channels=self.input_channels,
            out_channels=None,
            depth=len(self.num_filters),
            initial_features=num_filters[0],
            norm=norm,
        )

        self.prior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            norm=norm,
        )

        use_onehot = output_channels > 1
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            posterior=True,
            num_classes=output_channels,
            use_onehot=use_onehot,
            num_raters=num_raters,
            norm=norm,
        )

        self.fcomb = Fcomb(
            self.num_filters,
            self.latent_dim,
            self.output_channels,
            self.no_convs_fcomb,
        )

        if rl_swap:
            self._criterion = DiceLoss()
        elif output_channels == 1:
            self._criterion = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self._criterion = nn.CrossEntropyLoss(reduction="none")

        self.to(self.device)

    def _check_shape(self, patch: torch.Tensor) -> None:
        spatial_shape = patch.shape[2:]
        depth = len(self.num_filters)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for Probabilistic U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)

    def forward(self, patch: torch.Tensor, segm: Optional[torch.Tensor] = None) -> None:
        """Run the image through the UNet and build the prior latent space.

        Labels have shape (B, R, H, W). The joint posterior receives all R raters.
        """
        self._check_shape(patch)

        if segm is not None:
            self._check_labels(segm, patch)
            self.posterior_latent_space = self.posterior(patch, segm)

        self.prior_latent_space = self.prior(patch)
        self.unet_features = self.unet(patch)

    def _check_labels(self, segm: torch.Tensor, image: torch.Tensor) -> None:
        expected_shape = (image.shape[0], self.num_raters, *image.shape[2:])
        if segm.shape != expected_shape:
            raise ValueError(f"Expected labels with shape {expected_shape}, got {tuple(segm.shape)}.")

    def sample(self) -> torch.Tensor:
        """Sample a segmentation from the prior and decode it through the UNet feature map.

        Draws a latent vector z from the prior p(z|x) and passes it through fcomb together
        with the UNet features to produce one segmentation hypothesis. Calling this N times
        yields N diverse hypotheses for the same input image.
        """
        z_prior = self.prior_latent_space.sample()
        return self.fcomb(self.unet_features, z_prior)

    def reconstruct(
        self,
        use_posterior_mean: bool = False,
        calculate_posterior: bool = False,
        z_posterior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode a joint posterior sample into logits with shape (B, C, H, W).

        Args:
            use_posterior_mean: Use the posterior mean as z instead of sampling.
            calculate_posterior: Draw a fresh reparametrized sample from the posterior.
                Ignored when use_posterior_mean=True or z_posterior is provided.
            z_posterior: Pre-computed posterior sample to decode. Used directly when
                use_posterior_mean=False and calculate_posterior=False.
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.mean
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb(self.unet_features, z_posterior)

    def kl_divergence(
        self,
        analytic: bool = True,
        calculate_posterior: bool = False,
        z_posterior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL(posterior || prior).

        Args:
            analytic: If True, compute the KL in closed form. If False, estimate via
                a posterior sample (log q(z) - log p(z)).
            calculate_posterior: Draw a fresh reparametrized sample when analytic=False.
                Ignored when z_posterior is provided.
            z_posterior: Pre-computed posterior sample for the Monte Carlo estimate.
                Only used when analytic=False.
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(
        self,
        segm: torch.Tensor,
        consm: Optional[torch.Tensor] = None,
        analytic_kl: bool = True,
        reconstruct_posterior_mean: bool = False,
    ) -> torch.Tensor:
        """Compute the evidence lower bound -E[log p(y|x,z)] + beta * KL(q||p).

        One joint posterior sample supplies the reconstruction and KL terms for each image.
        Reconstruction loss averages over raters. The KL term contributes once per image.

        Args:
            segm: Binary labels or multiclass indices of shape (B, R, H, W).
            consm: Optional shared mask (B, 1, H, W) or per-rater mask (B, R, H, W).
                Applied as a multiplicative weight when consensus_masking=True.
            analytic_kl: If True, compute the KL divergence in closed form.
            reconstruct_posterior_mean: If True, decode the posterior mean instead of a sample.
        """

        self._check_labels(segm, self.unet_features)
        mask = consm if self.consensus_masking else None
        if mask is not None:
            shared_shape = (segm.shape[0], 1, *segm.shape[2:])
            if mask.shape != segm.shape and mask.shape != shared_shape:
                raise ValueError(f"Expected consensus mask with shape {shared_shape} or {tuple(segm.shape)}.")
            mask = mask.expand_as(segm)

        z_posterior = self.posterior_latent_space.rsample()

        kl_term = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
        )

        reconstruction = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean,
            calculate_posterior=False,
            z_posterior=z_posterior
        )
        losses = []
        for rater in range(self.num_raters):
            target = segm[:, rater:rater + 1]
            rater_mask = None if mask is None else mask[:, rater:rater + 1]
            losses.append(self._reconstruction_loss(reconstruction, target, rater_mask))
        reconstruction_loss = torch.stack(losses).mean()
        return -(reconstruction_loss + self.beta * kl_term)

    def _reconstruction_loss(
        self, reconstruction: torch.Tensor, segm: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        use_dice = self.rl_swap
        use_bce = isinstance(self._criterion, nn.BCEWithLogitsLoss)
        if use_dice:
            if self.output_channels == 1:
                prediction = reconstruction.sigmoid()
                target = segm.float()
            else:
                prediction = reconstruction.softmax(dim=1)
                target = F.one_hot(segm.squeeze(1).long(), self.output_channels).movedim(-1, 1).float()
            if mask is not None:
                # Mask probabilities before Dice reduces the spatial dimensions.
                prediction = prediction * mask
                target = target * mask
            reconstruction_loss = self._criterion(prediction, target)
            if mask is not None:
                reconstruction_loss = reconstruction_loss * mask.any()
        else:
            segm_t = segm.float() if use_bce else segm.squeeze(1).long()
            reconstruction_loss = self._criterion(reconstruction, segm_t)
            if mask is not None:
                reconstruction_loss = reconstruction_loss * (mask if use_bce else mask.squeeze(1))

        if use_dice:
            return reconstruction_loss

        # Sum over spatial dims, mean over batch - keeps loss scale independent of batch size.
        reconstruction_loss = reconstruction_loss.sum(dim=tuple(range(1, reconstruction_loss.dim()))).mean()
        return reconstruction_loss
