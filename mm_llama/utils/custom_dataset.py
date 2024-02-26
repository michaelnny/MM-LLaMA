from typing import Iterable, Dict, List, Iterator
import copy
import pickle
import json
import os
import math
import numpy as np
import random
import itertools

import torch
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset, DataLoader


def _compute_aggregated_stats(stats_dict: Dict) -> Dict:
    results = {}

    for k, v in stats_dict.items():
        if len(v) == 0:
            continue

        _mean = np.mean(v)
        _std = np.std(v)
        _min = np.min(v)
        _max = np.max(v)
        results[f'{k}_mean'] = round(_mean, 2)
        results[f'{k}_std'] = round(_std, 2)
        results[f'{k}_min'] = _min
        results[f'{k}_max'] = _max

    return results


class MMFineTuneDataset(Dataset):
    """Multi-modal fine-tune dataset, supports group by number of medias in a prompt"""

    def __init__(self, data_sources: Iterable[str], max_seq_len: int = 2048):
        """
        Args:
            data_sources: a list of string path to where to load the dataset.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
        """
        assert len(data_sources) > 0, f'Invalid data sources {data_sources!r}'
        assert max_seq_len >= 100

        super().__init__()

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len
        self.data = []

        # track statistics
        self.stats = {
            'prompt_lengths': [],
            'completion_lengths': [],
            'prompt_medias': [],
        }

        # Load datasets
        for source_file in data_sources:
            assert os.path.exists(source_file)
            samples = pickle.load(open(source_file, 'rb'))
            for item in samples:
                x, y, media_pos, media_hidden = item['prompt_tokens'], item['completion_tokens'], item['prompt_media_pos'], item['prompt_media_hidden']
                len_x, len_y = len(x), len(y)

                if len_x + len_y <= self.max_seq_len:
                    self.data.append((x, y, media_pos, media_hidden))
                    self.stats['prompt_lengths'].append(len_x)
                    self.stats['completion_lengths'].append(len_y)

        # Build groupby indices, we group by number of media inputs, mostly should be 1, but 0 for text only samples
        self.num_media_to_indices_map = {}
        for i, item in enumerate(self.data):
            x, y, media_pos, media_hidden = item
            num_media = len(media_pos) if media_pos is not None else 0
            self.stats['prompt_medias'].append(num_media)

            # group by number of medias per prompt
            if num_media not in self.num_media_to_indices_map:
                self.num_media_to_indices_map[num_media] = []
            self.num_media_to_indices_map[num_media].append(i)

        self.aggr_stats = _compute_aggregated_stats(self.stats)

    def __getitem__(self, idx):
        x, y, media_pos, media_hidden = self.data[idx]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        if media_pos is not None:
            media_pos = torch.tensor(media_pos, dtype=torch.long)
            media_hidden = torch.tensor(media_hidden, dtype=torch.float)

        return x, y, media_pos, media_hidden

    def __len__(self):
        return len(self.data)

    @property
    def group_indices(self) -> List[List[int]]:
        return list(self.num_media_to_indices_map.values())

    def get_metadata(self) -> Dict:
        return {
            'data_sources': self.data_sources,
            'num_samples': len(self),
            **self.aggr_stats,
        }


class GroupedBatchSampler(BatchSampler):
    """Batch sampler with group by attributes, where it takes in a list of pre-computed groupby indices for the dataset"""

    def __init__(self, group_indices: List[List[int]], batch_size: int, skip_minor_group: bool = True) -> None:
        assert batch_size >= 1
        assert len(group_indices) > 0

        self.batch_size = batch_size
        self.group_indices = group_indices

        # skip the group where number of indices is lesser than batch_size
        if skip_minor_group:
            self.groupby_subsets = [indices for indices in self.group_indices if len(indices) >= batch_size]
        else:
            self.groupby_subsets = [indices for indices in self.group_indices]
        assert len(self.groupby_subsets) > 0, 'No groups available to form custom batch sampler'

        num_samples = sum([len(x) for x in self.groupby_subsets])
        self.num_batches = math.ceil(num_samples / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        groups = range(len(self.groupby_subsets))
        for _ in range(self.num_batches):
            batch = []
            group = random.choice(groups)  # randomly select a group
            for _ in range(self.batch_size):  # sample N items indices from the same group
                random_idx = random.choice(self.groupby_subsets[group])
                batch.append(random_idx)
            yield batch

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':

    def custom_collate_fn(batch, pad_id: int = 0, max_seq_len: int = 128, full_pad: bool = True):
        """
        Custom collate function to pad the sequence to maximum length in the batch,
        and compute the loss mask for the batch.
        """
        batch_size = len(batch)

        max_batch_seq_len = max([len(item[0]) + len(item[1]) for item in batch])
        assert max_batch_seq_len <= max_seq_len

        if full_pad:
            max_batch_seq_len = max_seq_len

        # concatenate prompt, completion together
        batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)

        # loss mask where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
        loss_mask = torch.full((batch_size, max_batch_seq_len), 0, dtype=torch.long)

        media_pos_list = []
        media_features_list = []

        for i, (prompt, completion, prompt_media_pos, prompt_media_features) in enumerate(batch):
            # need prompt, completion lengths to compute loss mask
            prompt_len, completion_len = len(prompt), len(completion)
            seq_len = prompt_len + completion_len
            seq = torch.concat((prompt, completion), dim=0).type(torch.long)

            # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
            batch_sequences[i, :seq_len] = seq
            loss_mask[i, :prompt_len] = -1  # prompt tokens
            loss_mask[i, prompt_len : prompt_len + completion_len] = 1  # completion tokens

            # handle pre-computed media hidden position and features
            if prompt_media_pos is not None:
                media_pos_list.append(prompt_media_pos)
                media_features_list.append(prompt_media_features)

        x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_len - 1]
        y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_len - 1]

        # shift to right to align with y
        loss_mask = loss_mask[:, 1:]

        # concat prompt media position and features
        media_pos = None
        media_hidden = None
        if len(media_pos_list) > 0:
            media_pos = torch.stack(media_pos_list, dim=0)
            media_hidden = torch.stack(media_features_list, dim=0)

        return x, y, loss_mask, media_pos, media_hidden

    cuda_kwargs = {'collate_fn': custom_collate_fn, 'num_workers': 1, 'pin_memory': False}

    test_ds = MMFineTuneDataset(['./datasets/LLaVA_CC3M/validation.pkl'], 128)
    batch_size = 128
    gradient_accum_steps = 8

    num_train_iterations = 5

    # train_kwargs = {'batch_size': batch_size, 'shuffle': True, 'drop_last': True, 'sampler': None}

    # Create a DataLoader using the custom sampler
    batch_sampler = GroupedBatchSampler(test_ds, batch_size, drop_last=True)
    train_kwargs = {'batch_sampler': batch_sampler}

    train_kwargs.update(cuda_kwargs)
    dataloader = DataLoader(test_ds, **train_kwargs)

    previous_batches = []
    for i in range(num_train_iterations):
        print(f'Iteration {i}')

        batch = []
        for iter, (x, y, loss_mask, media_pos, media_hidden) in enumerate(dataloader):
            # if len(batch) > 0:
            #     print([torch.equal(x, x_tm1) for x_tm1 in batch])
            batch.append(x)

            if iter % gradient_accum_steps == 0:
                batch = torch.stack(batch, dim=0)

                if len(previous_batches) > 0:
                    print([torch.equal(batch, b_tm1) for b_tm1 in previous_batches])

                previous_batches.append(batch)

                batch = []
