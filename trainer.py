from typing import List, Tuple

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from simulator import Simulator
from smega2 import SMEGA2
from utils import AverageMeter, get_lr, set_lr


class Trainer:
    def __init__(self, model: Module, simulator: Simulator, train_loader: DataLoader, test_loader: DataLoader,
                 optimizer: SMEGA2, loss_fn, epochs: int, warmup_epochs: int, lr_schedule: List[int], lr_decay: float,
                 cuda: bool):
        self.model = model
        self.simulator = simulator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        lr = get_lr(optimizer)
        self.warmup_lr = np.linspace(lr / simulator.size, lr, len(self.train_loader) * warmup_epochs).tolist()
        self.lr_schedule = lr_schedule
        self.lr_decay = lr_decay
        self.cuda = cuda

    def run(self) -> None:
        for epoch in range(self.epochs):
            self.decay_lr(epoch)
            train_loss, train_acc = self.train_step(epoch)
            test_loss, test_acc = self.eval_step()
            wandb.log(dict(
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            ))

    def batch_step(self, data: Tensor, target: Tensor, loss_meter: AverageMeter, acc_meter: AverageMeter) -> Tensor:
        if self.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        bs = target.shape[0]
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss_meter.update(loss.item(), bs)  # sum up batch loss
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        acc_meter.update(100.0 * pred.eq(target.data.view_as(pred)).sum().item() / bs, bs)
        return loss

    def train_step(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        progress_bar = tqdm(self.train_loader)
        for data, target in progress_bar:
            self.simulator.load_next_worker()

            if self.warmup_lr:
                new_lr = self.warmup_lr.pop(0)
                set_lr(self.optimizer, new_lr)

            self.optimizer.zero_grad()

            loss = self.batch_step(data, target, loss_meter, acc_meter)
            loss.backward()

            self.simulator.load_master()
            self.optimizer.step()
            self.simulator.update_master(self.model.parameters())

            estimate = self.optimizer.estimate()
            self.simulator.update_worker(estimate)

            progress_bar.set_description(
                f"Epoch: {epoch}, Loss: {loss_meter.avg:.8f} Acc: {acc_meter.avg:.4f}")
        progress_bar.close()

        return loss_meter.avg, acc_meter.avg

    def eval_step(self) -> Tuple[float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                self.batch_step(data, target, loss_meter, acc_meter)
        return loss_meter.avg, acc_meter.avg

    def decay_lr(self, epoch: int) -> None:
        if epoch in self.lr_schedule:
            new_lr = get_lr(self.optimizer) * self.lr_decay
            set_lr(self.optimizer, new_lr)
