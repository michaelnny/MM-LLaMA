# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
Module for build instruct fine-tuning datasets.
"""

from typing import Tuple, List, Mapping, Text, Dict
import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import math
import functools
import json
import random
import pickle
import re
import time

import copy
import numpy as np
import torch


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.models.tokenizer import Tokenizer
from mm_llama.utils.file_helper import find_certain_files_under_dir, read_json_file, read_jsonl_file, count_words
from mm_llama.utils.prompt_builder import build_prompt_completion, Dialog
from mm_llama.utils.logger import create_logger

from mm_llama.models.imagebind.models.imagebind_model import imagebind_huge, ImageBindModel, ModalityType
from mm_llama.models.imagebind import data as imagebind_data

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    'role': 'system',
    'content': '',
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]


# ----------------------------------- helper functions -----------------------------------


def _split_and_save_datasets(
    datasets: List[dict],
    validation_ratio: float,
    train_output_file: str,
    val_output_file: str,
    meta_output_file: str,
    meta: dict,
) -> None:
    # split into train and validation datasets as dolly only have one single .json file
    random.shuffle(datasets)

    val_size = int(len(datasets) * validation_ratio)
    train_size = len(datasets) - val_size

    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, val_size])

    for data, out_file in zip((train_set, val_set), (train_output_file, val_output_file)):
        if len(data) > 0:
            logger.info(f'Saving {len(data)} processed data to "{out_file}" ...')
            pickle.dump(data, open(out_file, 'wb'))

    meta = {
        **meta,
        'num_train_samples': len(train_set),
        'num_validation_samples': len(val_set),
    }

    logger.info(f'Saving metadata to "{meta_output_file}" ...')

    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def process_dolly_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Dolly',
        'language': 'English',
        'home_page': 'https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm',
    },
) -> None:
    """Process dolly dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.info(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...')
        return

    if metadata is None:
        metadata = {}

    json_objs = read_jsonl_file(src_file)

    if json_objs is None:
        logger.info(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing dolly dataset...')
    datasets = []

    for item in json_objs:
        context = item['context'].strip()
        prompt = item['instruction'].strip()
        completion = item['response'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens, *_ = build_prompt_completion(dialog, tokenizer)

        assert all(r is not None for r in (prompt_tokens, completion_tokens))

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'prompt_media_pos': None, 'prompt_media_hidden': None})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed dolly dataset...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_alpaca_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'Alpaca_cleaned',
        'language': 'English',
        'home_page': 'https://github.com/gururise/AlpacaDataCleaned',
    },
) -> None:
    """Process alpaca dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    """

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.info(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...')
        return

    if metadata is None:
        metadata = {}

    json_objs = read_json_file(src_file)

    if json_objs is None:
        logger.info(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing alpaca dataset...')
    datasets = []

    for item in json_objs:
        context = item['input'].strip()
        prompt = item['instruction'].strip()
        completion = item['output'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens, *_ = build_prompt_completion(dialog, tokenizer)

        assert all(r is not None for r in (prompt_tokens, completion_tokens))

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'prompt_media_pos': None, 'prompt_media_hidden': None})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed alpaca dataset...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


@torch.no_grad
def process_llava_instruct_dataset(
    instruct_file: str,
    image_src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    imagebind_ckpt: str,
    batch_size: int = 512,
    validation_ratio: float = 0.05,
    max_samples: int = 0,
    overwrite_output: bool = False,
    is_coco: bool = True,
):
    """
    Process LLaVA instruction dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<IMG>>UNK<</IMG>> [/INST]", "completion": " {1st response} </s>"}

    In addition, include the following two properties:
        "prompt_media_pos": "numpy array contains the media hidden feature position index in the prompt tokens"
        "prompt_media_hidden": "numpy array contains the ImageBind pre-computed hidden features for the image"

    """
    assert os.path.exists(instruct_file) and os.path.isfile(instruct_file)
    assert os.path.exists(image_src_dir) and os.path.isdir(image_src_dir)
    assert os.path.exists(imagebind_ckpt) and os.path.isfile(imagebind_ckpt)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.info(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    imagebind_model = imagebind_huge(imagebind_ckpt)
    imagebind_model.eval()
    imagebind_model.to(device)

    if is_coco:
        metadata = {
            'name': 'LLaVA-COCO',
            'language': 'English',
            'home_page': 'https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md',
            'description': 'Multi-turn dialog dataset with image input from COCO dataset, generated by GPT-4',
        }
    else:
        metadata = {
            'name': 'LLaVA-CC-3M',
            'language': 'English',
            'home_page': 'https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md',
            'description': 'Single-turn dialog dataset with image input from CC-3M dataset, generated by GPT-4',
        }

    dialog_list = []

    with open(instruct_file, 'r', encoding='utf-8') as file:
        data = json.loads(file.read())

    logger.info('Preparing data for LLaMA-2 dialog structure...')
    for item in data:  # for each dialog
        if len(item['conversations']) < 2:
            continue

        image_path = os.path.join(image_src_dir, item['image'])
        if not os.path.exists(image_path):
            logger.info(f'Image file {image_path!r} does not exists')
            continue

        dialog = []
        for i, conversation in enumerate(item['conversations']):  # for each turn in current dialog
            role = 'user' if conversation['from'] == 'human' else 'assistant'
            content = conversation['value'].replace('<image>\n', '').replace('\n<image>', '').replace('</image>\n', '').strip()

            turn = {'role': role, 'content': content}
            if i == 0:
                turn['media'] = {'type': 'image', 'file_path': image_path, 'hidden_features': None}

            dialog.append(turn)

        dialog_list.append(dialog)

    if max_samples > 0 and len(dialog_list) > max_samples:
        random.shuffle(dialog_list)
        dialog_list = dialog_list[:max_samples]

    assert all([dialog[0]['media'] is not None and 'file_path' in dialog[0]['media'] for dialog in dialog_list])

    logger.info(f'Computing hidden features using ImageBind model for {len(dialog_list)} dialogs ...')

    num_batches = math.ceil(len(dialog_list) / batch_size)
    pbar = tqdm.tqdm(range(num_batches), colour='green', desc='batches')
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > len(dialog_list):
            end_idx = len(dialog_list)

        image_files = [item[0]['media']['file_path'] for item in dialog_list[start_idx:end_idx]]

        inputs = {
            ModalityType.VISION: imagebind_data.load_and_transform_vision_data(image_files, device),
        }

        embeddings = imagebind_model(inputs)
        hidden_features = embeddings[ModalityType.VISION].cpu()  # [batch_size, 1024]

        # set the hidden features property for the dialog
        for j in range(len(hidden_features)):
            idx = start_idx + j
            dialog_list[idx][0]['media']['hidden_features'] = hidden_features[j, ...].clone()  # [1024]

        pbar.update(1)

    pbar.close()

    del imagebind_model
    del inputs
    del embeddings
    del hidden_features
    torch.cuda.empty_cache()

    assert all([dialog[0]['media']['hidden_features'] is not None for dialog in dialog_list])
    logger.info('Building tokenized prompt:completion pairs ...')

    # Tokenize the text
    # IMPORTANT, the standard multiprocessing and ProcessPool and submit will cause the program to hang as the task is too big (~million)
    # One can try to split the data into sub-trunks, but the speed up is barely noticeable
    pbar = tqdm.tqdm(range(len(dialog_list)), colour='green', desc='dialogs')
    datasets = []
    for dialog in dialog_list:
        prompt_tokens, completion_tokens, media_pos, media_hidden = build_prompt_completion(dialog, tokenizer)
        assert all(r is not None for r in (prompt_tokens, completion_tokens, media_pos, media_hidden))
        datasets.append(
            {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'prompt_media_pos': media_pos,
                'prompt_media_hidden': media_hidden,
            }
        )
        pbar.update(1)

    pbar.close()

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of dict object for the prompt:completion pair'
    metadata['data_structure_details'] = {
        'prompt_tokens': 'tokenized prompt text, with a placeholder <<IMG>>UNK<</IMG>> token for the image embedding',
        'completion_tokens': 'tokenized completion text',
        'prompt_media_pos': 'numpy array contains the media hidden feature position index in the prompt tokens',
        'prompt_media_hidden': 'numpy array contains the ImageBind pre-computed hidden features for the image',
    }

    logger.info('Saving processed LLaVA instruct dataset...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


def process_single_video(args: Tuple[int, str], device, clip_duration, clips_per_video) -> Dict:
    index, video_file = args

    features = imagebind_data.load_and_transform_video_data([video_file], device=device, clip_duration=clip_duration, clips_per_video=clips_per_video)
    # remove batch dimension
    return (index, features.squeeze(0))


@torch.no_grad
def process_videochat_dataset(
    instruct_file: str,
    video_src_dir: str,
    output_dir: str,
    tokenizer: Tokenizer,
    imagebind_ckpt: str,
    batch_size: int = 512,
    clip_duration: int = 2,
    clips_per_video: int = 5,
    max_turns_per_dialog: int = 6,
    validation_ratio: float = 0.05,
    num_workers: int = 8,
    max_samples: int = 0,
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'VideoChat',
        'language': 'English',
        'home_page': 'https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data',
        'description': 'Chat dialog dataset with video input from WebVid-10M dataset, generated by GPT-4. Video source from https://maxbain.com/webvid-dataset/',
    },
):
    """
    Process LLaVA instruction dataset and save the tokenized prompt:completion pairs to .pkl format.

    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<VID>>UNK<</VID>> [/INST] {1st response} </s><s>[INST] {2nd user prompt} <<IMG>>\n?? \n<</IMG>> [/INST]", "completion": " {2nd response} </s>"}

    In addition, include the following two properties:
        "prompt_media_pos": "numpy array contains the media hidden feature position index in the prompt tokens"
        "prompt_media_hidden": "numpy array contains the ImageBind pre-computed hidden features for the image"

    """
    assert os.path.exists(instruct_file) and os.path.isfile(instruct_file)
    assert os.path.exists(video_src_dir) and os.path.isdir(video_src_dir)
    assert os.path.exists(imagebind_ckpt) and os.path.isfile(imagebind_ckpt)
    assert clip_duration >= 1
    assert clips_per_video >= 1
    assert max_turns_per_dialog >= 1
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.info(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting...')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    imagebind_model = imagebind_huge(imagebind_ckpt)
    imagebind_model.eval()
    imagebind_model.to(device)

    if metadata is None:
        metadata = {}

    dialog_list = []

    with open(instruct_file, 'r', encoding='utf-8') as file:
        data = json.loads(file.read())

    logger.info('Preparing data for LLaMA-2 dialog structure...')

    for item in data:  # for each dialog
        if len(item['QA']) < 1:
            continue

        # "050951_051000/1022713978.mp4" -> "1022713978.mp4"
        video_file = item['video'].split('/')[1]

        video_path = os.path.join(video_src_dir, video_file)
        if not os.path.exists(video_path):
            logger.info(f'Video file {video_path!r} does not exists')
            continue

        dialog = []
        for i, conversation in enumerate(item['QA']):  # for each turn in current dialog
            question_content = conversation['q']
            answer_content = conversation['a']

            turn_user = {'role': 'user', 'content': question_content}
            turn_assistant = {'role': 'assistant', 'content': answer_content}
            if i == 0:
                turn_user['media'] = {'type': 'video', 'file_path': video_path, 'hidden_features': None}

            dialog.append(turn_user)
            dialog.append(turn_assistant)

        # Break long conversions into subsets, this reuse the same video for multiple training samples
        if len(dialog) > max_turns_per_dialog:
            num_splits = len(dialog) // max_turns_per_dialog
            for i in range(0, num_splits, 6):
                start = i * max_turns_per_dialog
                end = start + max_turns_per_dialog if i < num_splits - 1 else len(dialog)
                subset = dialog[start:end]
                # insert the media to each subset
                if 'media' not in subset[0]:
                    subset[0]['media'] = copy.deepcopy(dialog[0]['media'])
                dialog_list.append(subset)
        else:
            dialog_list.append(dialog)

    if max_samples > 0 and len(dialog_list) > max_samples:
        random.shuffle(dialog_list)
        dialog_list = dialog_list[:max_samples]

    assert all([dialog[0]['media'] is not None and 'file_path' in dialog[0]['media'] for dialog in dialog_list])

    logger.info(f'Computing hidden features using ImageBind model for {len(dialog_list)} dialogs ...')

    num_batches = math.ceil(len(dialog_list) / batch_size)
    pbar = tqdm.tqdm(range(num_batches), colour='green', desc='batches')
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if end_idx > len(dialog_list):
            end_idx = len(dialog_list)

        video_files = [item[0]['media']['file_path'] for item in dialog_list[start_idx:end_idx]]

        # using multiprocessing to speed up video pre-processing
        args_list = [(i, video) for i, video in enumerate(video_files)]
        process_video = functools.partial(process_single_video, device='cpu', clip_duration=clip_duration, clips_per_video=clips_per_video)

        with mp.Pool(num_workers) as pool:
            results = list(pool.map(process_video, args_list))

        # Sort the results based on the index
        sorted_results = sorted(results, key=lambda x: x[0])

        # Extract only the processed results
        final_results = [result for _, result in sorted_results]

        stacked_videos = torch.stack(final_results, dim=0).to(device)

        inputs = {
            ModalityType.VISION: stacked_videos,
        }

        embeddings = imagebind_model(inputs)

        hidden_features = embeddings[ModalityType.VISION].cpu()  # [batch_size, 1024]

        # set the hidden features property for the dialog
        for j in range(len(hidden_features)):
            idx = start_idx + j
            dialog_list[idx][0]['media']['hidden_features'] = hidden_features[j, ...].clone()  # [1024]

        pbar.update(1)

    pbar.close()

    del imagebind_model
    del stacked_videos
    del inputs
    del embeddings
    del hidden_features
    torch.cuda.empty_cache()

    assert all([dialog[0]['media']['hidden_features'] is not None for dialog in dialog_list])
    logger.info('Building tokenized prompt:completion pairs ...')

    # Tokenize the text
    pbar = tqdm.tqdm(range(len(dialog_list)), colour='green', desc='dialogs')
    datasets = []
    for dialog in dialog_list:
        prompt_tokens, completion_tokens, media_pos, media_hidden = build_prompt_completion(dialog, tokenizer)
        assert all(r is not None for r in (prompt_tokens, completion_tokens, media_pos, media_hidden))
        datasets.append(
            {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'prompt_media_pos': media_pos,
                'prompt_media_hidden': media_hidden,
            }
        )
        pbar.update(1)

    pbar.close()

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of dict object for the prompt:completion pair'
    metadata['data_structure_details'] = {
        'prompt_tokens': 'tokenized prompt text, with a placeholder <<VID>>UNK<</VID>> tokens for the video embedding',
        'completion_tokens': 'tokenized completion text',
        'prompt_media_pos': 'numpy array contains the media hidden feature position index in the prompt tokens',
        'prompt_media_hidden': 'numpy array contains the ImageBind pre-computed hidden features for the video',
    }

    logger.info('Saving processed VideoChat instruct dataset...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )


if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    # Set multiprocessing start mode
    mp.set_start_method('spawn')

    tokenizer = Tokenizer(model_path='/home/michael/models/meta_llama2/tokenizer.model')

    # process_dolly_dataset(
    #     src_file='/home/michael/datasets/dolly_15k/databricks-dolly-15k.jsonl',
    #     output_dir='./datasets/dolly',
    #     tokenizer=tokenizer,
    # )

    # process_alpaca_dataset(
    #     src_file='/home/michael/datasets/alpaca_dataset/alpaca_cleaned.json',
    #     output_dir='./datasets/alpaca',
    #     tokenizer=tokenizer,
    # )

    # LLaVA CC-3M only has single-turn dialog, where COCO has multiple turn

    process_llava_instruct_dataset(
        instruct_file='/home/michael/datasets/LLaVA_CC3M/chat.json',
        image_src_dir='/home/michael/datasets/CC3M/images',
        output_dir='./datasets/LLaVA_CC3M',
        tokenizer=tokenizer,
        imagebind_ckpt='/home/michael/models/ImageBind/imagebind_huge.pth',
        batch_size=512,
        is_coco=False,
    )

    process_llava_instruct_dataset(
        instruct_file='/home/michael/datasets/LLaVA_COCO/llava_instruct_150k.json',
        image_src_dir='/home/michael/datasets/COCO/train2017',
        output_dir='./datasets/LLaVA_COCO',
        tokenizer=tokenizer,
        imagebind_ckpt='/home/michael/models/ImageBind/imagebind_huge.pth',
        batch_size=512,
        is_coco=True,
    )

    process_videochat_dataset(
        instruct_file='/home/michael/datasets/WebVid/videochat_instruct_11k.json',
        video_src_dir='/home/michael/datasets/WebVid/videos',
        output_dir='./datasets/VideoChat_WebVid',
        tokenizer=tokenizer,
        imagebind_ckpt='/home/michael/models/ImageBind/imagebind_huge.pth',
        batch_size=64,
        num_workers=24,
    )
