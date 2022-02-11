import argparse
import random
from functools import partial
from pprint import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import wandb

from datasets import cifar10, cifar100, imagenet, fake_imagenet
from pytorch_resnet_cifar10.resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from simulator import Simulator
from smega2 import SMEGA2
from trainer import Trainer
from worker_selection import GammaRandomWorkerSelection

torch.backends.cudnn.benchmark = True


DATASETS = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'imagenet': imagenet,
    'fake-imagenet': fake_imagenet,
}

CIFAR_MODELS = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56(),
    'resnet110': resnet110,
    'resnet1202': resnet1202,
}

IMAGENET_MODELS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Environment")

    train_parser = parser.add_argument_group("Train Parameters")
    train_parser.add_argument("--dataset", default="cifar10", metavar="DS",
                              help="train dataset (default: cifar10)",
                              choices=DATASETS.keys())
    train_parser.add_argument("--model", default="resnet20", metavar="MDL",
                              help="model to train (default: resnet20)",
                              choices=CIFAR_MODELS.keys() | IMAGENET_MODELS.keys())
    train_parser.add_argument("--epochs", type=int, default=160, metavar="E",
                              help="number of epochs to train (default: 10)")
    train_parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                              help="input batch size for training (default: 128)")
    train_parser.add_argument("--test-batch-size", type=int, default=128, metavar="BT",
                              help="input batch size for testing (default: 128)")
    train_parser.add_argument("--lr_decay", type=float, default=0.1, metavar="LD",
                              help="learning rate decay rate (default: 0.1)")
    train_parser.add_argument("--schedule", type=int, nargs="*", default=[80, 120],
                              help="learning rate is decayed at these epochs (default: [80, 120])")
    train_parser.add_argument("--warmup-epochs", type=int, default=5, metavar="WE",
                              help="number of warmup epochs (default: 5)")
    train_parser.add_argument("--no-cuda", action="store_true", default=False,
                              help="disables CUDA training (default: False)")
    train_parser.add_argument("--seed", type=int, default=None, metavar="S",
                              help="random seed (default: None)")

    simulator_parser = parser.add_argument_group("Simulator Parameters")
    simulator_parser.add_argument("--sim-size", type=int, default=8, metavar="N",
                                  help="number of workers to simulate (default: 8)")
    simulator_parser.add_argument("--sim-gamma-shape", type=float, default=100, metavar="GSH",
                                  help="gamma shape parameter (default: 100)")
    simulator_parser.add_argument("--sim-gamma-scale", type=float, default=None, metavar="GSC",
                                  help="gamma scale parameter (default: B / GSH)")

    optimizer_parser = parser.add_argument_group("Optimizer Parameters")
    optimizer_parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)")
    optimizer_parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                                  help="SGD momentum (default: 0.9)")
    optimizer_parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD",
                                  help="SGD weight decay (default: 1e-4)")
    optimizer_parser.add_argument("--naive", default=False, action="store_true",
                                  help="disables SMEGA2 advanced estimation")

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    random.seed(torch.initial_seed())

    if not args.sim_gamma_scale:
        args.sim_gamma_scale = args.batch_size / args.sim_gamma_shape

    pprint(vars(args))

    train_set, test_set = DATASETS[args.dataset]()
    data_loader = partial(DataLoader, num_workers=2, pin_memory=True)
    train_loader = data_loader(train_set, shuffle=True, batch_size=args.batch_size)
    test_loader = data_loader(test_set, shuffle=False, batch_size=args.test_batch_size)

    available_models = CIFAR_MODELS if 'cifar' in args.dataset else IMAGENET_MODELS
    try:
        model = available_models[args.model]()
    except KeyError as e:
        raise ValueError(f"{args.model} is not supported for {args.dataset}."
                         f" Please use one of {list(available_models.keys())}") from e
    if args.cuda:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    optimizer = SMEGA2(model.parameters(), lr=args.lr, momentum_cycle=args.sim_size, advanced=not args.naive,
                       momentum=args.momentum, weight_decay=args.weight_decay)
    assert len(optimizer.param_groups) == 1

    worker_selector = GammaRandomWorkerSelection(args.sim_size, args.sim_gamma_shape, args.sim_gamma_scale)

    simulator = Simulator(args.sim_size, worker_selector, model)

    trainer = Trainer(
        model,
        simulator,
        train_loader,
        test_loader,
        optimizer,
        loss_fn,
        args.epochs,
        args.warmup_epochs,
        args.schedule,
        args.lr_decay,
        args.cuda,
    )

    wandb.init(config=args, anonymous="allow")
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
