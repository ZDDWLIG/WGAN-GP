# GAN and WGAN-GP
GAN and WGAN-gp implemented with pytorch and test on the mnist dataset

# Installation
Clone this repo:

```
  git clone https://github.com/ZDDWLIG/WGAN-GP.git
  cd path/to/GAN-pytorch-master
```

Use conda to manage the python environment:

```
  conda create -n GAN python=3.8
  conda activate GAN
  pip install -r requirements.txt
```
# Train

For example, if you want to train WGAN-gp , then run：

```
  python WGAN-GP.py
```


# MNIST test results

## Epoch 0

![epoch0](https://github.com/ZDDWLIG/GAN-pytorch/blob/master/image/mnist_epoch%3D0.png)

## Epoch 40

![epoch40](https://github.com/ZDDWLIG/GAN-pytorch/blob/master/image/mnist_epoch%3D40.png)

## Epoch 160

![epoch160](https://github.com/ZDDWLIG/GAN-pytorch/blob/master/image/mnist_epoch%3D160.png)

## Loss

![Loss](https://github.com/ZDDWLIG/GAN-pytorch/blob/master/image/loss.png)
