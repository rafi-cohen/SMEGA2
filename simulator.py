from typing import Iterable, List

import torch
from torch import Tensor
from torch.nn import Module

Tensors = Iterable[Tensor]


class Simulator:
    def __init__(self, sim_size: int, worker_selector: Iterable[int], model: Module):
        self.size = sim_size
        self.worker_selector_iterator = iter(worker_selector)
        self.model = model
        self.master_weights = self.clone_weights(model.parameters())
        self.worker_weights = [self.clone_weights(model.parameters()) for _ in range(sim_size)]
        self.current_rank = next(self.worker_selector_iterator)

    def clone_weights(self, params: Tensors) -> List[Tensor]:
        return [p.clone() for p in params]

    def set_weights(self, source: Tensors, target: Tensors) -> None:
        with torch.no_grad():
            for s, t in zip(source, target):
                t.set_(s.detach().clone())

    def load_worker(self, worker_rank: int) -> None:
        self.set_weights(source=self.worker_weights[worker_rank], target=self.model.parameters())

    def load_next_worker(self) -> None:
        self.current_rank = next(self.worker_selector_iterator)
        self.load_worker(self.current_rank)

    def update_worker(self, weights: Tensors) -> None:
        self.set_weights(source=weights, target=self.worker_weights[self.current_rank])

    def load_master(self) -> None:
        self.set_weights(source=self.master_weights, target=self.model.parameters())

    def update_master(self, weights: Tensors):
        self.set_weights(source=weights, target=self.master_weights)
