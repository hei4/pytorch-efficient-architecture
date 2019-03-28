# pytorch-efficient-architecture
Efficient Convolutional Neural Networks with PyTorch



## Network architectures

Every stage has 4 blocks.

![architecture](./images/architecture.png)

## Blocks
- plain type

  - BatchNorm - ReLU - Conv3x3 - BatchNorm - ReLU - Conv3x3

- residual type

- bottlenecked residual type

  - [He, Kaiming et al. “Deep Residual Learning for Image Recognition.” *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (2016): 770-778.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

- resnext type

  - [Xie, Saining et al. “Aggregated Residual Transformations for Deep Neural Networks.” *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (2017): 5987-5995.](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)

- xception type

  - [Chollet, François. “Xception: Deep Learning with Depthwise Separable Convolutions.” *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (2017): 1800-1807.](http://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)

- dense type

  - [Huang, Gao et al. “Densely Connected Convolutional Networks.” *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (2017): 2261-2269.](http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html)

- mobile type

  - [Howard, Andrew G. et al. “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.” CoRR abs/1704.04861 (2017): n. pag.](https://arxiv.org/abs/1704.04861)

- shuffle type

  - [Zhang, Xiangyu et al. “ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (2018): 6848-6856.](https://arxiv.org/abs/1707.01083)

  


## Experiments

### MNIST
|training|validation|
|---|---|
|![MNIST train acc](images/mnist_train_acc.png "MNIST training accuracy")|![MNIST valid acc](images/mnist_valid_acc.png "MNIST validation accuracy")|
|![MNIST train loss](images/mnist_train_loss.png "MNIST training loss")|![MNIST valid loss](images/mnist_valid_loss.png "MNIST validation loss")|
|![MNIST train time](images/mnist_train_time.png "MNIST training inference time")|![MNIST valid time](images/mnist_valid_time.png "MNIST validation inference time")|

![MNIST file size](images/mnist_size.png "MNIST File size of trained models")

![MNIST cpu time](images/mnist_cpu_time.png "MNIST Inference time by CPU")

![MNIST gpu time](images/mnist_gpu_time.png "MNIST Inference time by GPU")

### CIFAR-10
|training|validation|
|---|---|
|![CIFAR-10 train acc](images/cifar10_train_acc.png "CIFAR-10 training accuracy")|![CIFAR-10 valid acc](images/cifar10_valid_acc.png "CIFAR-10 validation accuracy")|
|![CIFAR-10 train loss](images/cifar10_train_loss.png "CIFAR-10 training loss")|![CIFAR-10 valid loss](images/cifar10_valid_loss.png "CIFAR-10 validation loss")|
|![CIFAR-10 train time](images/cifar10_train_time.png "CIFAR-10 training inference time")|![CIFAR-10 valid time](images/cifar10_valid_time.png "CIFAR-10 validation inference time")|

![CIFAR-10 file size](images/cifar10_size.png "CIFAR-10 File size of trained models")

![CIFAR-10 cpu time](images/cifar10_cpu_time.png "CIFAR-10 Inference time by CPU")

![CIFAR-10 gpu time](images/cifar10_gpu_time.png "CIFAR-10 Inference time by GPU")


### STL-10
|training|validation|
|---|---|
|![STL-10 train acc](images/stl10_train_acc.png "STL-10 training accuracy")|![STL-10 valid acc](images/stl10_valid_acc.png "STL-10 validation accuracy")|
|![STL-10 train loss](images/stl10_train_loss.png "STL-10 training loss")|![STL-10 valid loss](images/stl10_valid_loss.png "STL-10 validation loss")|
|![STL-10 train time](images/stl10_train_time.png "STL-10 training inference time")|![STL-10 valid time](images/stl10_valid_time.png "STL-10 validation inference time")|

![STL-10 file size](images/stl10_size.png "STL-10 File size of trained models")

![STL-10 cpu time](images/stl10_cpu_time.png "STL-10 Inference time by CPU")

![STL-10 gpu time](images/stl10_gpu_time.png "STL-10 Inference time by GPU")


### Food-101
UNDER CONSTRUCTION.
