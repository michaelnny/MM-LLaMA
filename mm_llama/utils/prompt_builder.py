# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Build prompt-completion pairs using LLaMA-2 predefined chat style format.

We rewrite the original code to make it easier to understand, we can also use the code to build fine-tuning samples.
"""
from typing import Tuple, List, Mapping, Text, Union, Literal, Optional, TypedDict
import os

import torch
import numpy as np

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.models.tokenizer import Tokenizer
from mm_llama.models.imagebind.models.imagebind_model import imagebind_huge, ImageBindModel, ModalityType
from mm_llama.models.imagebind import data as imagebind_data

Role = Literal['system', 'user', 'assistant']

# Modality media type, only support image and video
MediaType = Literal['image', 'video']


class Media(TypedDict):
    type: MediaType
    file_path: str
    hidden_features: Optional[torch.Tensor]


class Message(TypedDict):
    role: Role
    content: str
    media: Optional[Media]


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


# Check if a type is supported
supported_types = set(MediaType.__args__)


def is_supported_media_type(media_type: str) -> bool:
    return media_type and media_type in supported_types


Dialog = List[Message]

B_INST, E_INST = '[INST]', '[/INST]'
B_SYS, E_SYS = '<<SYS>>\n', '\n<</SYS>>\n\n'

# Multi-Modal special tokens
B_IMG, E_IMG = '<<IMG>>', '<</IMG>>'
B_VID, E_VID = '<<VID>>', '<</VID>>'


def maybe_add_system_prompt(dialog: Dialog) -> Dialog:
    """Try to insert an empty system prompt at the beginning to make code consistent."""
    assert dialog is not None and len(dialog) > 0

    if dialog[0]['role'] != 'system':
        dialog = [
            {
                'role': 'system',
                'content': '',
            }
        ] + dialog

    return dialog


@torch.no_grad
def build_prompt_completion(
    dialog: Dialog, tokenizer: Tokenizer, imagebind_model: Optional[ImageBindModel] = None
) -> Tuple[List[int], List[int], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Build prompt and completion pair following the Meta llama-2 format.

    Note we only build the training target completion if the last role in the dialog is 'assistant'.

    Here are some examples of the format that llama-2 uses (before apply tokenization), note we inserted BOS and EOS into the example for completeness:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<IMG>>UNK<</IMG>> [/INST]", "completion": " {1st response} </s>"}
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<VID>>UNK<</VID>> [/INST] {1st response} </s><s>[INST] {2nd user prompt} <<IMG>>\n?? \n<</IMG>> [/INST]", "completion": " {2nd response} </s>"}

    """

    assert dialog is not None and len(dialog) >= 1

    dialog = maybe_add_system_prompt(dialog)

    assert len(dialog) >= 2

    assert dialog[0]['role'] == 'system' and all([msg['role'] == 'user' for msg in dialog[1::2]]) and all([msg['role'] == 'assistant' for msg in dialog[2::2]]), (
        "model only supports 'system', 'user' and 'assistant' roles, " "starting with 'system', then 'user' and alternating (u/a/u/a/u ...)"
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if imagebind_model:
        imagebind_model.eval()
        imagebind_model.to(device=device)

    # store user-prompt:answer pairs so we can later add BOS, EOS tokens to each pair
    prompts = []
    for i in range(1, len(dialog), 2):  # skip the first one since it's system prompt
        prompt_start = ''
        prompt_end = ''
        prompt_media_hidden_features = None
        prompt = dialog[i]

        # handle multi-modal media special tokens
        media_type_begin, media_type_end = '', ''
        hidden_features = None
        if 'media' in prompt and 'type' in prompt['media']:
            media_type = prompt['media']['type']
            media_path = prompt['media']['file_path']

            if not is_supported_media_type(media_type):
                raise RuntimeError(f'Invalid media type {media_type!r}')

            if not media_path or not os.path.exists(media_path):
                raise RuntimeError(f'Invalid media file path {media_path!r}')

            inputs = None
            modality_type = None
            if media_type == 'image':
                modality_type = ModalityType.VISION
                media_type_begin, media_type_end = B_IMG, E_IMG

                if 'hidden_features' in prompt['media']:
                    # use pre-computed hidden features
                    hidden_features = prompt['media']['hidden_features']

                else:
                    hidden_features = None
                    inputs = {
                        ModalityType.VISION: imagebind_data.load_and_transform_vision_data([media_path], device),
                    }

            elif media_type == 'video':
                modality_type = ModalityType.VISION
                media_type_begin, media_type_end = B_VID, E_VID

                if 'hidden_features' in prompt['media']:
                    # use pre-computed hidden features
                    hidden_features = prompt['media']['hidden_features']
                else:
                    hidden_features = None
                    inputs = {
                        ModalityType.VISION: imagebind_data.load_and_transform_video_data([media_path], device),
                    }

            if hidden_features is None and inputs is not None:
                if not imagebind_model:
                    raise RuntimeError('ImageBind model is needed to compute hidden features')

                embeddings = imagebind_model(inputs)
                hidden_features = embeddings[modality_type]  # [1, 1024]

            if hidden_features is not None:
                if len(hidden_features.shape) == 2:
                    hidden_features = hidden_features.squeeze(0)
                hidden_features = hidden_features.cpu().numpy()  # [1024,]

        # handle system prompt
        if i == 1:
            # note Meta llama-2 inserts the system prompt inside the first user prompt
            # as in this format: [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]
            sys_prompt = dialog[0]['content'].strip()
            if len(sys_prompt) > 0:
                sys_prompt = B_SYS + sys_prompt + E_SYS

            # example format: [INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<IMG>><</IMG>> [/INST]
            prompt_start = f"{B_INST} {sys_prompt}{(prompt['content']).strip()} {media_type_begin}"
        else:
            prompt_start = f"{B_INST} {(prompt['content']).strip()} {media_type_begin}"

        prompt_end = f'{media_type_end} {E_INST}'

        # make previous response as part of the prompt
        # here we skip the last response by the assistant, since it's used for building the target for training
        if i + 1 < len(dialog) - 1 and dialog[i + 1]['role'] == 'assistant':
            answer = dialog[i + 1]
            prompt_end += f" {(answer['content']).strip()}"

        prompts.append((prompt_start, prompt_end, hidden_features))

    # Concatenate and tokenize the full prompt, note llama-2, we add BOS and EOS for every pair of user-prompt:answer, except for the last user-prompt
    hidden_pos_list = []
    hidden_features_list = []
    prompt_tokens = []
    for i, (prompt_start, prompt_end, hidden_features) in enumerate(prompts):
        start_tokens = tokenizer.encode(prompt_start, bos=True, eos=False)
        prompt_tokens.extend(start_tokens)
        if hidden_features is not None:
            # insert a placeholder for modality embed, we only need one token as ImageBind always output a embed with shape (1024, ) for different inputs
            prompt_tokens.append(tokenizer.unk_id)
            hidden_pos_list.append(len(prompt_tokens) - 1)  # minus 1 because indexing starts from 0
            hidden_features_list.append(hidden_features)

        end_tokens = tokenizer.encode(prompt_end, bos=False, eos=i < len(prompts) - 1)
        prompt_tokens.extend(end_tokens)

    # build completion tokens for training
    completion_tokens = None
    if dialog[-1]['role'] == 'assistant':
        answer = dialog[-1]
        target = f" {(answer['content']).strip()}"
        completion_tokens = tokenizer.encode(target, bos=False, eos=True)

    if len(hidden_pos_list) > 0:
        prompt_media_pos = np.stack(hidden_pos_list, axis=0)
        prompt_media_hidden_features = np.stack(hidden_features_list, axis=0)
    else:
        prompt_media_pos = None
        prompt_media_hidden_features = None

    return (prompt_tokens, completion_tokens, prompt_media_pos, prompt_media_hidden_features)


if __name__ == '__main__':
    tokenizer = Tokenizer(model_path='./meta_checkpoints/LLaMA-2/tokenizer.model')
    imagebind_model = imagebind_huge('./checkpoints/imagebind/imagebind_huge.pth')
    example_dialogs = [
        [
            {'role': 'user', 'content': 'Solve 1+37.'},
            {'role': 'assistant', 'content': '38'},
        ],
        [
            {'role': 'user', 'content': 'Tell me a joke about a dog.', 'media': {'type': 'image', 'file_path': '.assets/bird_image.jpg'}},
        ],
        [
            {
                'role': 'system',
                'content': 'You are a very clever and funny agent, make people laugh is your natural job.',
            },
            {'role': 'user', 'content': 'Tell me a joke about a cat playing some toy car.', 'media': {'type': 'video', 'file_path': '.assets/bird_video.mp4'}},
        ],
        [
            {'role': 'user', 'content': 'I am going to Paris, what should I see?', 'media': {'type': 'video', 'file_path': '.assets/bird_video.mp4'}},
            {'role': 'assistant', 'content': 'You should go to The Eiffel Tower.'},
            {'role': 'user', 'content': "What's so special about it?", 'media': {'type': 'image', 'file_path': '.assets/bird_image.jpg'}},
        ],
        [
            {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
            {'role': 'assistant', 'content': 'You should go to The Eiffel Tower.'},
            {'role': 'user', 'content': "What's so special about it?"},
            {'role': 'assistant', 'content': "Just go there and you'll find out."},
            {'role': 'user', 'content': 'Can you be more specific?'},
            {'role': 'assistant', 'content': 'What do you mean be more specific?'},
        ],
    ]

    for dialog in example_dialogs:
        prompt_tokens, completion_tokens, media_pos_idx, prompt_media_hidden_features = build_prompt_completion(
            dialog,
            tokenizer,
            imagebind_model,
        )

        print(f'Prompt: {tokenizer.decode(prompt_tokens)}')

        if completion_tokens is not None:
            print(f'Completion: {tokenizer.decode(completion_tokens)}')

        if media_pos_idx is not None:
            print(media_pos_idx.shape)
        if prompt_media_hidden_features is not None:
            print(prompt_media_hidden_features.shape)

        print('\n\n')
