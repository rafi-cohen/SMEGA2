from typing import Type, TypeVar, Generic

from torch.optim import Optimizer

T = TypeVar('T')


class AverageMeter(Generic[T]):
    def __init__(self, data_type: Type[T] = float):
        self._data_type = data_type
        self._avg: T = data_type(0)
        self._count: int = 0

    def reset(self) -> None:
        self._avg = self._data_type(0)
        self._count = 0

    def update(self, val: T, n: int = 1) -> None:
        self._count += n
        self._avg += (n / self._count) * (val - self._avg)

    @property
    def avg(self) -> T:
        return self._avg


def get_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer: Optimizer, new_lr: float) -> None:
    optimizer.param_groups[0]["lr"] = new_lr
