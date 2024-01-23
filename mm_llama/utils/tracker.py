from typing import Dict
import numpy as np

import torch
import torch.distributed as dist


class StatsTracker:
    def __init__(self, distributed: bool = False):
        self.metrics = torch.zeros(5).to('cuda' if distributed else 'cpu')
        self.c = 0

        self.distributed = distributed

    def update(self, loss: torch.Tensor, num_accurate: int, num_samples: int):
        metrics = self.metrics

        metrics[0] += loss.item()  # sum up batch loss
        metrics[1] += np.exp(loss.item())  # sum up perplexity
        metrics[2] += 1  # increase number of micro batches
        metrics[3] += num_accurate  # sum up number of accurate prediction tokens
        metrics[4] += num_samples  # sum up number of tokens

        self.c += 1

    def reset(self) -> None:
        self.metrics = torch.zeros(5)
        self.c = 0

    def get_dict(self) -> Dict:
        if self.c == 0:
            return {}

        metrics = self.metrics

        if self.distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        loss = metrics[0] / metrics[2]
        perplexity = metrics[1] / metrics[2]
        accuracy = 100 * metrics[3] / metrics[4]

        return {'loss': loss.item(), 'accuracy': accuracy.item(), 'perplexity': perplexity.item()}
