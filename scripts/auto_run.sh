python train.py --epoch 40 --dataset mnist --block plain
python train.py --epoch 40 --dataset mnist --block residual
python train.py --epoch 40 --dataset mnist --block residual_bottleneck
python train.py --epoch 40 --dataset mnist --block resnext
python train.py --epoch 40 --dataset mnist --block xception
python train.py --epoch 40 --dataset mnist --block clc

python train.py --epoch 60 --dataset cifar10 --block plain
python train.py --epoch 60 --dataset cifar10 --block residual
python train.py --epoch 60 --dataset cifar10 --block residual_bottleneck
python train.py --epoch 60 --dataset cifar10 --block resnext
python train.py --epoch 60 --dataset cifar10 --block xception
python train.py --epoch 60 --dataset cifar10 --block clc

python train.py --epoch 80 --dataset stl10 --block plain
python train.py --epoch 80 --dataset stl10 --block residual
python train.py --epoch 80 --dataset stl10 --block residual_bottleneck
python train.py --epoch 80 --dataset stl10 --block resnext
python train.py --epoch 80 --dataset stl10 --block xception
python train.py --epoch 80 --dataset stl10 --block clc

python train.py --epoch 100 --dataset food101 --block plain
python train.py --epoch 100 --dataset food101 --block residual
python train.py --epoch 100 --dataset food101 --block residual_bottleneck
python train.py --epoch 100 --dataset food101 --block resnext
python train.py --epoch 100 --dataset food101 --block xception
python train.py --epoch 100 --dataset food101 --block clc
