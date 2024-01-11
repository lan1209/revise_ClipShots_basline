# 修改后的deepsbd，原版链接参考后文

直接运行main_cla.py即可，train.py里的参数修改后会覆盖main_cla.py

主要修改的地方为：

（1）models/deepSBD.py

将网络结构修改为[deepSBD](https://arxiv.org/abs/1705.08214) 中介绍的模式

（2）opts.py

根据实际情况修改部分参数，解决了源代码的报错情况

（3）split_data.py

由于数据集过大，解压移动时造成了部分数据的损失，因此从原数据集中随机选择了100个视频，所有以choose开头的文件都是后面随机选择的。数据不对应会导致程序无法正常运行。
由于只是用于实验，该代码还不完善，如果使用完整数据集则不需要运行此代码。

（4）eval_res.py

调整了输出格式使其规划化。


# DeepSBD for ClipShots
This repository contains our implementation of [deepSBD](https://arxiv.org/abs/1705.08214) for [ClipShots](https://github.com/Tangshitao/ClipShots). The code is modified from [here](https://github.com/kenshohara/3D-ResNets-PyTorch).

## Introduction
We implement deepSBD in this repository. There are 2 backbones that can be selected, including the original Alexnet-like and ResNet-18 introduced in our [paper](https://arxiv.org/pdf/1808.04234.pdf).

## Resources
1. The trained model for Alexnet-like backbone. [BaiduYun](https://pan.baidu.com/s/16q3CNuUhLAGkm21PPOqUSg), [Google Drive](https://drive.google.com/open?id=145NCxLhgdrKPIYm-qgp1SRYU_GFmzxxX)
2. The trained model for ResNet-18 backbone. [BaiduYun](https://pan.baidu.com/s/1Bx2uVVQOuEnTxdBBGV3uCQ), [Google Drive](https://drive.google.com/file/d/1CVqxAp17OOBmNq9_jgEdaoDbrmK5Bmog/view?usp=sharing)
3. The pretrained [model](https://drive.google.com/open?id=10h_axdnkjupEDYe-OiUzm5ALX8w5DX_5) for ResNet-18 backbone.

## Training
Please refer to `train.sh`

## Testing
Add '--no_train' options to `train.sh`.
