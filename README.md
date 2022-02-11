# SMEGA2
SMEGA2: Distributed Deep Learning Using a Single Momentum Buffer

# Installation
```
git clone --recurse-submodules https://github.com/rafi-cohen/SMEGA2.git
cd SMEGA2
pip install -r requirements.txt
```

# Usage
```text
usage: main.py [-h] [--dataset DS] [--model MDL] [--epochs E] [--batch-size B] [--test-batch-size BT] [--lr_decay LD] [--schedule [SCHEDULE [SCHEDULE ...]]] [--warmup-epochs WE]
               [--no-cuda] [--seed S] [--sim-size N] [--sim-gamma-shape GSH] [--sim-gamma-scale GSC] [--lr LR] [--momentum M] [--weight-decay WD] [--naive]

PyTorch Environment

optional arguments:
  -h, --help            show this help message and exit

Train Parameters:
  --dataset DS          train dataset: cifar10, cifar100, imagenet, fake-
                        imagenet (default: cifar10)
  --model MDL           model to train: resnet56, resnet34, resnet44,
                        resnet110, resnet152, resnet18, resnet101, resnet32,
                        resnet50, resnet1202, resnet20 (default: resnet20)
  --epochs E            number of epochs to train (default: 10)
  --batch-size B        input batch size for training (default: 128)
  --test-batch-size BT  input batch size for testing (default: 128)
  --lr_decay LD         learning rate decay rate (default: 0.1)
  --schedule [SCHEDULE [SCHEDULE ...]]
                        learning rate is decayed at these epochs (default: [80, 120])
  --warmup-epochs WE    number of warmup epochs (default: 5)
  --no-cuda             disables CUDA training (default: False)
  --seed S              random seed (default: None)

Simulator Parameters:
  --sim-size N          number of workers to simulate (default: 8)
  --sim-gamma-shape GSH
                        gamma shape parameter (default: 100)
  --sim-gamma-scale GSC
                        gamma scale parameter (default: B / GSH)

Optimizer Parameters:
  --lr LR               learning rate (default: 0.1)
  --momentum M          SGD momentum (default: 0.9)
  --weight-decay WD     SGD weight decay (default: 1e-4)
  --naive               disables SMEGA2 advanced estimation
```

# Viewing Results
The results are tracked using [Weights & Biases](https://wandb.ai/). You can run the code, see a W&B dashboard, and visualize results without needing to create a Weights & Biases account first.

A link to the dashboard will be printed to the terminal before training starts.
