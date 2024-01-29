# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Run supervised fine-tuning (STF) using QLoRA, starting with a pretrained model."""
import os
import functools
import argparse
from typing import Tuple, Union, Mapping, Text, Any, Dict
import tqdm
import random

import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.models.llama_lora import Transformer, LoraModelArgs
from mm_llama.models.tokenizer import Tokenizer
from mm_llama.models.lora import mark_only_lora_as_trainable

from mm_llama.configs.pretrain import config as PretrainCfg
from mm_llama.configs.finetune import config as FinetuneCfg
from mm_llama.utils.custom_dataset import MMFineTuneDataset, GroupedBatchSampler

from mm_llama.utils.schedule import CosineDecayWithWarmupLRScheduler
from mm_llama.utils.train_helper import create_optimizer, compute_num_trainable_params, get_grad_norm_local
from mm_llama.utils.checkpoint import create_lora_checkpoint
from mm_llama.utils.logger import create_logger
from mm_llama.utils.tracker import StatsTracker


RunArgsType = Union[PretrainCfg, FinetuneCfg]

logger = create_logger()


def clear_gpu_cache():
    torch.cuda.empty_cache()


def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, prompt_loss_weight: float, completion_loss_weight: float) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]

    B, T, *_ = logits.shape
    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    assert not torch.any(torch.isnan(losses))
    losses = losses.view(B, T)
    assert losses.shape == mask.shape

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, prompt_loss_weight).masked_fill(mask == 1, completion_loss_weight)
    losses *= weights  # [batch_size, seq_len]
    losses = losses.mean(1)  # [batch_size]
    return losses


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert targets.shape == mask.shape  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights.detach()
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)


def train_step(model: Transformer, batch: Tuple[torch.Tensor], scaler: torch.cuda.amp.GradScaler, loss_scale: float, tracker: StatsTracker, loss_fn: Any) -> None:
    """Run a single training step, where we do a forward + backward passes, but do no update parameters"""
    x, y, loss_mask, media_pos, media_hidden = batch
    x, y, loss_mask = (
        x.to('cuda', non_blocking=True),
        y.to('cuda', non_blocking=True),
        loss_mask.to('cuda', non_blocking=True),
    )

    if media_pos is not None:
        media_pos = media_pos.to('cuda', non_blocking=True)
        media_hidden = media_hidden.to('cuda', non_blocking=True)

    output = model(x, prompt_media_pos=media_pos, prompt_media_hidden=media_hidden)
    losses = loss_fn(output, y, loss_mask)
    loss = losses.mean()
    scaled_loss = loss * loss_scale

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    num_acc, num_samples = compute_metrics(output.detach(), y.detach(), loss_mask.detach())
    tracker.update(losses.detach(), num_acc, num_samples)


def update_step(
    model: Transformer,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler = None,
) -> None:
    """Run a single parameter update step"""
    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch


@torch.no_grad()
def run_validation_steps(model: Transformer, loader: DataLoader, steps: int, tracker: StatsTracker, loss_fn: Any) -> None:
    """Run M validation steps"""

    tracker.reset()
    inner_pbar = tqdm.tqdm(range(steps), colour='green', desc='Validation steps')

    for i, (x, y, loss_mask, media_pos, media_hidden) in enumerate(loader):
        x, y, loss_mask = (
            x.to('cuda', non_blocking=True),
            y.to('cuda', non_blocking=True),
            loss_mask.to('cuda', non_blocking=True),
        )

        if media_pos is not None:
            media_pos = media_pos.to('cuda', non_blocking=True)
            media_hidden = media_hidden.to('cuda', non_blocking=True)

        output = model(x, prompt_media_pos=media_pos, prompt_media_hidden=media_hidden)

        losses = loss_fn(output, y, loss_mask)
        num_acc, num_samples = compute_metrics(output.detach(), y.detach(), loss_mask.detach())
        tracker.update(losses.detach(), num_acc, num_samples)

        if inner_pbar is not None:
            inner_pbar.update(1)

        if i >= steps:
            break

    inner_pbar.close()


def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
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


def main(args: RunArgsType):
    assert args.num_epochs >= 1
    assert args.train_batch_size >= 1
    assert args.gradient_accum_steps >= 1
    assert 0 < args.loss_scale <= 1
    assert args.log_interval >= 1
    assert args.val_interval >= 1
    assert args.val_steps >= 1

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    if not os.path.exists(args.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint {args.pretrain_ckpt_file!r}, aborting ...')

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')

    tokenizer = Tokenizer(args.tokenizer_file)

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=args.max_seq_len,
        full_pad=args.full_pad,
    )

    loss_fn = functools.partial(
        compute_finetune_loss,
        prompt_loss_weight=args.prompt_loss_weight,
        completion_loss_weight=args.completion_loss_weight,
    )

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': args.dataloader_workers,
        'pin_memory': False,
    }

    train_dataset = MMFineTuneDataset(data_sources=args.train_datasources, max_seq_len=args.max_seq_len)
    train_kwargs = {'batch_size': args.train_batch_size, 'sampler': None, 'shuffle': True}
    # train_kwargs = {'batch_sampler': GroupedBatchSampler(group_indices=train_dataset.group_indices, batch_size=args.train_batch_size)}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger.info(f'Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create validation dataset
    val_loader = None
    if args.val_interval > 0:
        val_dataset = MMFineTuneDataset(data_sources=args.val_datasources, max_seq_len=args.max_seq_len)
        val_kwargs = {'batch_size': args.val_batch_size, 'sampler': None, 'shuffle': True}
        # val_kwargs = {'batch_sampler': GroupedBatchSampler(group_indices=val_dataset.group_indices, batch_size=args.val_batch_size)}
        val_kwargs.update(cuda_kwargs)
        val_loader = DataLoader(val_dataset, **val_kwargs)
        logger.info(f'Validation dataset metadata:\n{val_dataset.get_metadata()}')

    batch_size = int(args.train_batch_size * args.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // args.gradient_accum_steps
    max_train_steps = steps_per_epoch * args.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger.info('Initializing model and optimizer ...')

    torch.cuda.set_device('cuda:0')
    clear_gpu_cache()

    compute_dtype = torch.float32
    scaler = None
    if args.mixed_precision:
        if torch.version.cuda and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
    else:
        logger.info('Training in float32 mode, make sure you have enough GPU RAM')

    model_args = LoraModelArgs.from_model_type(
        model_type=args.model_type,
        # LoRA configurations
        lora_r=args.lora_r,
        lora_scaling=args.lora_scaling,
        lora_dropout=args.lora_dropout,
        # LoRA trainable layers
        lora_attn_query=args.lora_attn_query,
        lora_attn_key=args.lora_attn_key,
        lora_attn_value=args.lora_attn_value,
        lora_attn_proj=args.lora_attn_proj,
        lora_attn_mlp=args.lora_attn_mlp,
        lora_lm_head=args.lora_lm_head,
        # Quantization configurations
        quant_4bit=args.quant_4bit,
        quant_lora_4bit=args.quant_lora_4bit,
        quant_4bit_double=args.quant_4bit_double,
        quant_4bit_type=args.quant_4bit_type,
        quant_compute_dtype=compute_dtype,
        # Regular configurations
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dropout=args.embed_dropout,
        attn_dropout=args.attn_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights not contained in checkpoint state
    if os.path.exists(args.pretrain_ckpt_file):
        logger.info(f'Loading pretrained checkpoint {args.pretrain_ckpt_file!r} ...')
        model_state = torch.load(args.pretrain_ckpt_file)
        model.load_state_dict(model_state, strict=False)
        del model_state

    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        if 'norm' in name:  # for better performance, always use full precision for normalization layers
            module = module.to(dtype=torch.float32)
        else:
            module = module.to(dtype=compute_dtype)

    # make sure LLM alignment layers are triable
    additional_layers = ['llm_align_proj', 'llm_align_norm']
    mark_only_lora_as_trainable(model, args.train_bias, additional_layers)

    # This is where the weights quantization happens
    # when we move the model to cuda, the bnb.nn.Params4bit.cuda() method is called,
    # and the weights is quantized using bnb.functional.quantize_4bit
    model = model.to('cuda')

    # seems not so helpful in terms of speed improvement
    if args.compile_model:
        logger.info('compile model using torch.compile() ...')
        model = torch.compile(model)

    logger.info('Initializing optimizer ...')

    num_trainable, num_frozen = compute_num_trainable_params(model)
    logger.info(f'Number of trainable parameters: {num_trainable:,}')
    logger.info(f'Number of frozen parameters: {num_frozen:,}')

    optimizer = create_optimizer(
        model=model,
        lr=args.init_lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        betas=args.adam_betas,
        fused=args.adam_fused,
        paged_adamw=args.use_paged_adamw,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=int(args.warmup_ratio * max_train_steps),
        max_decay_steps=max_train_steps,
    )

    # --------------- Start Training ---------------

    create_ckpt_func = functools.partial(create_lora_checkpoint, train_bias=args.train_bias, additional_layers=additional_layers)

    tb_writer = None
    inner_pbar = tqdm.tqdm(range(max_train_steps), colour='blue', desc='Training steps')
    best_val_accuracy = 0.0
    train_steps = 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(args.log_dir, args.model_type))

    train_tracker = StatsTracker()
    val_tracker = StatsTracker()

    logger.info(f'Starting to run {args.num_epochs} training epochs, total of {max_train_steps} steps, with batch size {batch_size}')

    for epoch in range(1, args.num_epochs + 1):  # for each epoch
        logger.info(f'Start epoch {epoch}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()

        for iter, batch in enumerate(train_loader):  # for each batch in current epoch
            train_step(model, batch, scaler, args.loss_scale, train_tracker, loss_fn)

            if iter % args.gradient_accum_steps == 0:
                grad_norm = get_grad_norm_local(model)
                update_step(model, optimizer, scheduler, args.grad_clip, scaler)
                inner_pbar.update(1)
                train_steps += 1

                train_stats = train_tracker.get_dict(reset=True)

                # logging training statistics
                if train_steps % args.log_interval == 0:
                    train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
                    train_stats['grad_norm'] = grad_norm.item()
                    log_statistics(tb_writer, train_steps, train_stats, True)

                # regular checkpointing
                if args.ckpt_interval > 0 and (train_steps % args.ckpt_interval == 0 or train_steps == max_train_steps):
                    create_ckpt_func(model=model, full_path=os.path.join(args.ckpt_dir, f'lora_{args.model_type}-steps-{train_steps}.pth'))

                # validation steps
                if args.val_steps > 0 and (args.val_interval > 0 and train_steps % args.val_interval == 0 or train_steps == max_train_steps):
                    model.eval()
                    run_validation_steps(model, val_loader, args.val_steps, val_tracker, loss_fn)
                    model.train()

                    val_stats = val_tracker.get_dict(reset=True)
                    log_statistics(tb_writer, train_steps, val_stats, False)

                    # save best model
                    if val_stats['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_stats['accuracy']
                        logger.info(f'New best validation accuracy: {val_stats["accuracy"]:.2f}')
                        create_ckpt_func(model=model, full_path=os.path.join(args.ckpt_dir, f'lora_{args.model_type}-best.pth'))

    # show some training stats.
    logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')


def log_statistics(tb_writer: SummaryWriter, train_steps: int, stats: Dict, is_training: bool) -> None:
    logger.info(f'Training steps {train_steps}, is validation run: {not is_training}')
    logger.info(stats)

    if tb_writer is not None:
        tb_tag = 'train' if is_training else 'val'
        for k, v in stats.items():
            tb_writer.add_scalar(f'{tb_tag}/{k}', v, train_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        help='Run pre-training (stage 1) or fine-tuning (stage 2) stages',
        type=int,
        default=1,
        nargs='?',
    )
    args = parser.parse_args()

    if args.stage == 1:
        logger.info('Run training stage 1: MM-to-LLM alignment projection pre-training')
        run_args = PretrainCfg
    elif args.stage == 2:
        logger.info('Run training stage 2: MM-to-LLM jointed fine-tuning with LoRA')
        run_args = FinetuneCfg
    else:
        raise RuntimeError(f'Invalid stage argument {args.stage}')

    torch.manual_seed(run_args.seed)
    np.random.seed(run_args.seed)
    random.seed(run_args.seed)

    torch.set_float32_matmul_precision('high')

    main(run_args)
