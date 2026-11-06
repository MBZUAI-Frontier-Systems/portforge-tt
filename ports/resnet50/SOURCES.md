# Start from public evidence

Read CONTRACT.md first. Target: `timm/resnet50.tv_in1k` at revision
`78f3ecfdb38e06d9b8397f662e7ab8fee96026fa` (bsd-3-clause): the ResNet-50 of
He et al. (2016) with torchvision's ImageNet-1k weights (`IMAGENET1K_V1`),
packaged for timm. About 25.6M parameters; input 3x224x224.

- [He et al., Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385):
  the bottleneck design, BatchNorm placement and projection shortcuts.
- [timm model card for `resnet50.tv_in1k`](https://huggingface.co/timm/resnet50.tv_in1k):
  the pinned `config.json` (input size, bilinear resize, crop fraction 0.875,
  mean and std). The repository also ships `pytorch_model.bin`; it is not part
  of the pinned manifest.
- [torchvision `resnet50`](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html):
  `ResNet50_Weights.IMAGENET1K_V1`, reported at 76.13% top-1 and 92.86% top-5.
  The oracle's held-out top-1 should be close to that.
- [torchvision ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
  (BSD-3-Clause): the module the oracle instantiates (`resnet50(weights=None)`,
  then `load_state_dict(strict=True)`), with the v1.5 stride on the 3x3
  convolution.
- TTNN: the installed `ttnn` package's docstrings and the op unit tests under
  `/opt/tt-metal/tests/ttnn` in the sandbox; [docs.tenstorrent.com](https://docs.tenstorrent.com/)
  when web access is on. Confirm each operation's signature and layout
  requirements in the installed version before relying on it.
- [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k): the
  validation split the private corpus is drawn from (gated; not in this pack).
