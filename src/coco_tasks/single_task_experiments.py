import os
from typing import List, Union, Tuple

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from coco_tasks.settings import TB_ROOT
from coco_tasks.single_task_datasets import CocoTasksTest, CocoTasksGT, CocoTasksTestGT



class OverfitSampler(Sampler):
    def __init__(self, main_source, indices):
        super().__init__(main_source)
        self.main_source = main_source
        self.indices = indices

        main_source_len = len(self.main_source)

        how_many = int(round(main_source_len / len(self.indices)))
        self.to_iter_from = []
        for _ in range(how_many):
            self.to_iter_from.extend(self.indices)

    def __iter__(self):
        return iter(self.to_iter_from)

    def __len__(self):
        return len(self.main_source)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


