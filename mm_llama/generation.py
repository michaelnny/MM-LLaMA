# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os

import time
from typing import List, Literal, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.models.llama import ModelArgs, Transformer
from mm_llama.models.imagebind.models.imagebind_model import ImageBindModel
from mm_llama.models.tokenizer import Tokenizer
from mm_llama.utils.prompt_builder import (
    Message,
    Dialog,
    ChatPrediction,
    CompletionPrediction,
    build_prompt_completion,
)


class Llama:
    @staticmethod
    def build(
        ckpt_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
        seed: int = 1,
    ) -> 'Llama':
        if not os.path.exists(ckpt_path):
            raise ValueError(f'Checkpoint file {ckpt_path!r} does not exist, aborting ...')
        ckpt_dir = os.path.dirname(ckpt_path)

        params_path = os.path.join(ckpt_dir, 'params.json')
        if not os.path.exists(params_path):
            raise ValueError(f'Can not find model metadata file {params_path!r}, aborting ...')

        print(f'Starting to load model checkpoints {ckpt_path!r} ...')

        torch.manual_seed(seed)

        # Set these causes ImageBind model fail when processing videos
        # torch.set_default_device(device)
        # torch.set_default_dtype(torch.float16)

        t0 = time.time()
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        with open(params_path, 'r') as f:
            params = json.loads(f.read())

        try:
            del params['max_seq_len']
            del params['max_batch_size']
            del params['use_cache']
            del params['gradient_checkpointing']
        except Exception:
            pass

        model_args: ModelArgs = ModelArgs(
            **params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_cache=True,
        )

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        print(f'Model checkpoint loaded in {time.time() - t0:.2f} seconds')

        print(f'Starting to load tokenizer checkpoint {tokenizer_path!r} ...')
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size

        for params in model.parameters():
            params.requires_grad = False
        model = model.eval()

        # For support running ImageBind model and it's processors, since we can't set default device and tensor type
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        for name, module in model.named_modules():
            if 'norm' in name:  # for better performance, always use full precision for normalization layers
                module = module.to(dtype=torch.float32)
            else:
                module = module.to(dtype=compute_dtype)
        model = model.to(device)

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        media_pos: Optional[np.ndarray],
        media_features: Optional[np.ndarray],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> List[List[int]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz == 1  # due to handling of media position and attention cache, we only support batch size of 1

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device='cuda')
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')

        media_pos = torch.stack([torch.from_numpy(d) for d in media_pos]).to('cuda', dtype=torch.long) if media_pos is not None and len(media_pos) > 0 else None
        media_features = (
            torch.stack([torch.from_numpy(d) for d in media_features]).to('cuda', dtype=torch.float) if media_features is not None and len(media_features) > 0 else None
        )

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device='cuda')
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(
                tokens[:, prev_pos:cur_pos],
                prev_pos,
                media_pos if prev_pos == 0 else None,
                media_features if prev_pos == 0 else None,
            )
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None

            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]

            out_tokens.append(toks)

        return out_tokens

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        imagebind_model: ImageBindModel = None,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        for dialog in dialogs:
            current_prompt_tokens, _, current_media_pos, current_media_features = build_prompt_completion(dialog, self.tokenizer, imagebind_model)

            generation_token = self.generate(
                prompt_tokens=[current_prompt_tokens],
                media_pos=[current_media_pos] if current_media_pos is not None else None,
                media_features=[current_media_features] if current_media_features is not None else None,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            yield {'generation': {'role': 'assistant', 'content': self.tokenizer.decode(generation_token[0])}}


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
