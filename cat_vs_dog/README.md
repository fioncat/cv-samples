# Udacity毕业项目 Cat Vs Dog

使用迁移学习解决Kaggle竞赛猫狗大战并达到top50成绩.

## 依赖

- Python 3.6
- TensorFlow 1.10+
- Keras
- numpy
- matplotlib

## 数据集

Kaggle猫狗大战数据集下载: [Kaggle Dog Vs Cat](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).

下载后请解压并放置于你电脑中的任意位置,请记住这个位置.

## 提取特征

本项目在训练前需要提取训练和测试数据的特征.主要提取下面这几个特征:

- Inception V3
- Xception
- ResNet50
- VGG 16

详情见[extract_features.ipynb](extract_features.ipynb).

这一过程最花时间,建议使用GPU提取,在如下配置的机器下:

- CPU: Intel(R) Core i7-8750H 2.20GHz
- RAM: 16GB
- GPU: NVIDIA(R) GeForce GTX 1060 6GB

软件环境为:

- OS: Ubuntu 16.04 LTS
- CUDA: 9.0
- cuDNN: 7.4
- GPU Driver: 384.130

提取图像花费的时间如下:

图像个数|网络|时间
:-:|:-:|:-:
25000|Inception V3|11.2分钟
25000|Xception|10分钟
25000|ResNet50|11分钟
25000|VGG 16|7.7分钟
12500|Inception V3|5.9分钟
12500|Xception|5.4分钟
12500|ResNet50|5.8分钟
12500|VGG 16 |4分钟

## 训练

在提取特征之后,请在[train.ipynb](train.ipynb)下进行训练.

## 测试

训练结束后,需要生成Kaggle的提交结果.这在[test.ipynb](test.ipynb)完成.

我在自己的机器下的结果保存在[results/](results)下.
