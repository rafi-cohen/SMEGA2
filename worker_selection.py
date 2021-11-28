import heapq
from typing import List, Tuple, Iterator

import numpy as np


class GammaRandomWorkerSelection:
    def __init__(self, size: int, gamma_shape: float, gamma_scale: float):
        self.order: List[Tuple[float, int]] = []
        self.shape = gamma_shape
        self.scale = gamma_scale
        for i in range(size):
            rand = np.random.gamma(self.shape, self.scale, 1)
            heapq.heappush(self.order, (rand, i))

    def __iter__(self) -> Iterator[int]:
        while True:
            x = heapq.heappop(self.order)
            y = x[0] + np.random.gamma(self.shape, self.scale, 1)
            heapq.heappush(self.order, (y, x[1]))
            yield int(x[1])
