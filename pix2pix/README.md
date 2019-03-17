# 利用GAN实现Pix2Pix应用

**本项目由TensorFlow完成.**

这个应用来自TensorFlow官方示例. 地址[pix2pix_eager](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb).

原始应用使用的是Eager, 并且代码全部集成在一个notebook中. 我这里使用传统的Session实现, 代码封装在py文件中.

代码和官方的差不多, 我自己加了一些中文注释.

目录说明:

- [pix2pix_io.py](./pix2pix_io.py): 实现facedes数据集的读写.
- [pix2pix_model.py](./pix2pix_model.py): 实现GAN网络.
- [pix2pix.py](./pix2pix.py): 实现网络训练.
- [pix2pix.ipynb](./pix2pix.ipynb): 展示训练.
