# RDN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797v2).

## Table of contents

- [RDN-PyTorch](#rdn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train RDN model](#train-rdn-model)
        - [Resume train RDN model](#resume-train-rdn-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Residual Dense Network for Image Super-Resolution](#residual-dense-network-for-image-super-resolution)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

modify the `config.py`

- line 31: `model_arch_name` change to `rdn_small_x4`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `test`.
- line 40: `exp_name` change to `test_RDN_small_x4-DIV2K`.
- line 88: `model_weights_path` change to `./results/pretrained_models/RDN_small_x4-DIV2K-543022e7.pth.tar`.
-

```bash
python3 test.py
```

### Train RDN model

modify the `config.py`

- line 31: `model_arch_name` change to `rdn_small_x4`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `train`.
- line 40: `exp_name` change to `RDN_small_x4-DIV2K`.

```bash
python3 train.py
```

### Resume train RDN model

modify the `config.py`

- line 31: `model_arch_name` change to `rdn_small_x4`.
- line 37: `upscale_factor` change to `4`.
- line 39: `mode` change to `train`.
- line 40: `exp_name` change to `RDN_small_x4-DIV2K`.
- line 57: `resume_model_weights_path` change to `./results/RDN_small_x4-DIV2K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/abs/1802.08797v2](https://arxiv.org/abs/1802.08797v2)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

|  Method   | Scale |          Set5 (PSNR/SSIM)           |          Set14 (PSNR/SSIM)          | 
|:---------:|:-----:|:-----------------------------------:|:-----------------------------------:|
| RDN_small |   2   | 38.24(**38.11**)/0.9614(**0.9615**) | 34.01(**33.70**)/0.9212(**0.9185**) | 
| RDN_small |   3   | 34.71(**33.99**)/0.9296(**0.9240**) | 30.57(**30.06**)/0.8468(**0.8384**) |
| RDN_small |   4   | 32.47(**32.34**)/0.8990(**0.8977**) | 28.81(**28.71**)/0.7871(**0.7855**) | 

```bash
# Download `RDN_small_x4-DIV2K-543022e7.pth.tar` weights to `./results/pretrained_models/RDN_small_x4-DIV2K-543022e7.pth.tar`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="480" height="312" src="figure/119082_lr.png"/></span>

Output:

<span align="center"><img width="480" height="312" src="figure/119082_sr.png"/></span>

```text
Build `rdn_small_x4` model successfully.
Load `rdn_small_x4` model weights `./results/pretrained_models/RDN_small_x4-DIV2K-543022e7.pth.tar` successfully.
SR image save to `./figure/baboon_lr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Residual Dense Network for Image Super-Resolution

_Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, Yun Fu_ <br>

**Abstract** <br>
A very deep convolutional neural network (CNN) has recently achieved great success for image super-resolution (SR) and
offered hierarchical features as well. However, most deep CNN based SR models do not make full use of the hierarchical
features from the original low-resolution (LR) images, thereby achieving relatively-low performance. In this paper, we
propose a novel residual dense network (RDN) to address this problem in image SR. We fully exploit the hierarchical
features from all the convolutional layers. Specifically, we propose residual dense block (RDB) to extract abundant
local features via dense connected convolutional layers. RDB further allows direct connections from the state of
preceding RDB to all the layers of current RDB, leading to a contiguous memory (CM) mechanism. Local feature fusion in
RDB is then used to adaptively learn more effective features from preceding and current local features and stabilizes
the training of wider network. After fully obtaining dense local features, we use global feature fusion to jointly and
adaptively learn global hierarchical features in a holistic way. Extensive experiments on benchmark datasets with
different degradation models show that our RDN achieves favorable performance against state-of-the-art methods.

[[Paper]](https://arxiv.org/abs/1802.08797v2) [[Code]](https://github.com/jaewon-lee-b/rdn)

```bibtex
@inproceedings{zhang2018residual,
  title={Residual dense network for image super-resolution},
  author={Zhang, Yulun and Tian, Yapeng and Kong, Yu and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2472--2481},
  year={2018}
}
```
