# Diffusion Denoised Smoothing

This is the codebase for [(Certified!!) Adversarial Robustness for Free!](https://arxiv.org/abs/2206.10550).

This repository is based on [locuslab/smoothing](https://github.com/locuslab/smoothing), [openai/improved-diffusion](https://github.com/openai/improved-diffusion) and [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

---
We show how to achieve state-of-the-art certified adversarial robustness to 2-norm bounded perturbations by relying exclusively on off-the-shelf pretrained models.
<p align="center">
<img src="algo.png" width=100% height=100% 
class="center">
</p>

## Setup
Create an new conda virtual environment:
```
conda create -n diffusion_smoothing python=3.8 -y
conda activate diffusion_smoothing
```
Install Pytorch, torchvision following official instructions. For example:
```
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
```
Clone this repo and install the dependencies:
```
git clone https://github.com/ethz-privsec/diffusion_denoised_smoothing.git
pip install timm transformers statsmodels
```


We use these class-unconditional diffusion models from these repos:  
[CIFAR-10](https://github.com/openai/improved-diffusion): Unconditional CIFAR-10 with `L_hybrid` objective.  
[ImageNet](https://github.com/openai/guided-diffusion): Uncondtional 256x256 diffusion.  
Remember to download these model checkpoints in the corresponding directory.

## Evaluation
We give example evaluation command to run certification on CIFAR-10 and ImageNet.
```
# CIFAR-10
python cifar10/certify.py \
--sigma 1.00 --skip 1 --N0 100 --N 100000 --batch_size 200 \
--outfile [file to store certification results]
```

```
# ImageNet
python imagenet/certify.py \
--sigma 1.00 --skip 50 --N0 100 --N 10000 --batch_size 200 \
--outfile [file to store certification results]
```