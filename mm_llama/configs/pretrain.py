# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
from dataclasses import dataclass


@dataclass
class config:
    """Pre-training MM-to-LLM alignment projection layers"""

    # model type definition, the details (number of layers, heads etc.) are defined in model.py
    model_type: str = '7B'  # 7B, 13B, 70B
    max_seq_len: int = 512

    pretrain_ckpt_file: str = '/home/michael/models/meta_llama2/llama-2-7b-chat/consolidated.pth'  # load pretrained checkpoint
    tokenizer_file: str = '/home/michael/models/meta_llama2/tokenizer.model'  # load tokenizer model

    # datasets
    train_datasources: Tuple[str] = ('./datasets/LLaVA_CC3M/train.pkl',)
    val_datasources: Tuple[str] = ('./datasets/LLaVA_CC3M/validation.pkl',)
    dataloader_workers: int = 1

    # if true, always pad the sequence to max_seq_len instead of current maximum length in the batch
    # this is helpful when starting out and try to found the hyperparameter (e.g batch size, maximum sequence length)
    # so we may sooner found out CUDA out of memory error, rather than hours into the training process
    full_pad: bool = False

    # training and validation loops
    num_epochs: int = 3
    train_batch_size: int = 8
    # accumulate gradients, where for step, the batch size is = train_batch_size x gradient_accum_steps
    gradient_accum_steps: int = 16
    val_interval: int = 200
    val_batch_size: int = 30
    val_steps: int = 40
    log_interval: int = 5  # log training metrics (loss, accuracy)
    ckpt_interval: int = 200  # save model checkpoints every N training steps

    # LoRA configuration, During pre-training, all layers in LLM are frozen except the LLM alignment projection layer
    lora_r: int = 0
    lora_scaling: float = 1.0  # set the LoRA scaling, by default 1.0 no scaling
    lora_dropout: float = 0.0

    lora_attn_query: bool = False  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = False  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention projection layer
    lora_attn_mlp: bool = False  # train Attention MLP block
    lora_lm_head: bool = False  # train model output head

    train_bias: str = 'none'  # none, lora_only, all

    # Quantization
    quant_4bit: bool = False  # quantize frozen linear layer
    quant_lora_4bit: bool = False  # quantize LoRA linear layer
    quant_4bit_double: bool = False  # double quantize
    quant_4bit_type: str = 'nf4'  # only supports 'fp4' or 'nf4'

    # learning rate
    init_lr: float = 2e-5  # initial learning rate
    max_lr: float = 2e-4  # max learning rate after warm up
    min_lr: float = 2e-5  # min learning rate after decay
    warmup_ratio: float = 0.02

    # prompt is less important than completion
    prompt_loss_weight: float = 0.0
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    use_paged_adamw: bool = False
    weight_decay: float = 0.00
    adam_betas: Tuple = (0.9, 0.95)
    adam_eps: float = 1e-8
    adam_fused: bool = True  # only applicable if not using bitsandbytes optimizer
    grad_clip: float = 10.0

    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    llm_align_dropout: float = 0.1

    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    compile_model: bool = False  # not working with QLoRA

    # others
    seed: int = 127
    log_dir: str = './logs/pretrain'  # save logs and traces
    ckpt_dir: str = './checkpoints/pretrain'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces, be careful as the logs will grow very fast
