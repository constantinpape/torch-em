# Probabilistic U-Net with multiple raters

Set `num_raters` to the number of annotations per image. Pass images with shape
`(B, input_channels, H, W)` and labels with shape `(B, num_raters, H, W)`.
Keep the rater order consistent across images. All images must have the configured
number of annotations.

For binary segmentation, set `output_channels=1`. Each rater supplies a foreground
mask with values between zero and one. For multiclass segmentation, set
`output_channels` to the total number of classes, including background. Each rater
supplies one integer class ID per pixel, from zero to `output_channels - 1`.
Do not pass one-hot labels: the model converts each rater's class map internally.

```python
from torch_em.model import ProbabilisticUNet
from torch_em.self_training.loss import ProbabilisticUNetLoss, ProbabilisticUNetLossAndMetric

model = ProbabilisticUNet(
    input_channels=1, output_channels=4, num_raters=3, device="cpu"
)

# images: (B, 1, H, W), labels: (B, 3, H, W), integer class IDs 0 through 3.
loss = ProbabilisticUNetLoss()(model, images, labels)
loss.backward()

loss, metric = ProbabilisticUNetLossAndMetric()(model, images, labels)

# Inference needs only the images, without rater annotations.
model(images)
logits = model.sample()  # (B, 4, H, W)
probabilities = logits.softmax(dim=1)
```

For three binary raters, use `output_channels=1, num_raters=3` with the same label
shape. Apply sigmoid to sampled logits instead of softmax.

The posterior receives all raters jointly. Binary labels contribute `num_raters`
channels. Multiclass labels contribute `num_raters * output_channels` one-hot
channels, grouped by rater. The model compares one joint reconstruction against
each annotation and averages the reconstruction losses. The KL term contributes
once per image. This supports BCE, cross-entropy and Dice reconstruction losses.

Validation averages prior probabilities, evaluates the metric against each rater,
then averages the metrics. The default Dice metric receives one-hot multiclass
targets. Custom metrics receive one rater's original label map per call.

With `consensus_masking=True`, pass `label_filter` to either loss wrapper.
Use shape `(B, 1, H, W)` for a shared mask or `(B, num_raters, H, W)` for separate
masks. Zero-mask pixels contribute no reconstruction gradient. An empty rater mask
contributes zero reconstruction loss; the average still includes all configured
raters. The KL term and parameter regularization remain active.
