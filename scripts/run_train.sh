python train.py --epoch 40 --dataset mnist --block plain
python train.py --epoch 40 --dataset mnist --block residual
python train.py --epoch 40 --dataset mnist --block residual_bottleneck
python train.py --epoch 40 --dataset mnist --block resnext
python train.py --epoch 40 --dataset mnist --block xception
python train.py --epoch 40 --dataset mnist --block dense
python train.py --epoch 40 --dataset mnist --block mobile_v1
python train.py --epoch 40 --dataset mnist --block mobile_v2
python train.py --epoch 40 --dataset mnist --block shuffle

python train.py --epoch 60 --dataset cifar10 --block plain
python train.py --epoch 60 --dataset cifar10 --block residual
python train.py --epoch 60 --dataset cifar10 --block residual_bottleneck
python train.py --epoch 60 --dataset cifar10 --block resnext
python train.py --epoch 60 --dataset cifar10 --block xception
python train.py --epoch 60 --dataset cifar10 --block dense
python train.py --epoch 60 --dataset cifar10 --block mobile_v1
python train.py --epoch 60 --dataset cifar10 --block mobile_v2
python train.py --epoch 60 --dataset cifar10 --block shuffle

python train.py --epoch 80 --dataset stl10 --block plain
python train.py --epoch 80 --dataset stl10 --block residual
python train.py --epoch 80 --dataset stl10 --block residual_bottleneck
python train.py --epoch 80 --dataset stl10 --block resnext
python train.py --epoch 80 --dataset stl10 --block xception
python train.py --epoch 80 --dataset stl10 --block dense
python train.py --epoch 80 --dataset stl10 --block mobile_v1
python train.py --epoch 80 --dataset stl10 --block mobile_v2
python train.py --epoch 80 --dataset stl10 --block shuffle

# python train.py --epoch 100 --dataset food101 --block plain
# python train.py --epoch 100 --dataset food101 --block residual
# python train.py --epoch 100 --dataset food101 --block residual_bottleneck
# python train.py --epoch 100 --dataset food101 --block resnext
# python train.py --epoch 100 --dataset food101 --block xception
# python train.py --epoch 100 --dataset food101 --block dense
# python train.py --epoch 100 --dataset food101 --block mobile_v1
# python train.py --epoch 100 --dataset food101 --block clc
