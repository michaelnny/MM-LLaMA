# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from mm_llama.generation import Llama
from mm_llama.models.imagebind.models.imagebind_model import imagebind_huge


def main(
    ckpt_path: str,
    tokenizer_path: str,
    imagebind_ckpt_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 16,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device='cuda',
    )

    # Meta fine-tuned chat model
    dialogs = [
        # [
        #     {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
        #     {
        #         'role': 'assistant',
        #         'content': """\
        # Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:
        # 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
        # 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
        # 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
        # These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        #     },
        #     {'role': 'user', 'content': 'What is so great about #1?'},
        # ],
        # [
        #     {'role': 'user', 'content': 'What is the meaning of life?'},
        # ],
        # [
        #     {'role': 'user', 'content': 'Explain what is the theory of relativity.'},
        # ],
        # [
        #     {'role': 'user', 'content': 'Explain moon landing to a 8 years old kid.'},
        # ],
        # [
        #     {'role': 'user', 'content': 'Describe what is in the picture.', 'media': {'type': 'image', 'file_path': '.assets/dog_image.jpg'}},
        # ],
        # [
        #     {'role': 'user', 'content': 'Describe what is in the picture.', 'media': {'type': 'image', 'file_path': '.assets/car_image.jpg'}},
        # ],
        # [
        #     {'role': 'user', 'content': 'Describe what is in the picture.', 'media': {'type': 'image', 'file_path': '.assets/bird_image.jpg'}},
        # ],
        [
            {'role': 'user', 'content': 'Describe what is in the video.', 'media': {'type': 'video', 'file_path': '.assets/car_video.mp4'}},
        ],
        # [
        #     {'role': 'user', 'content': 'Describe what is in the video.', 'media': {'type': 'video', 'file_path': '.assets/dog_video.mp4'}},
        # ],
    ]

    imagebind_model = imagebind_huge(imagebind_ckpt_path)
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        imagebind_model=imagebind_model,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"---> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print('\n==================================\n')


if __name__ == '__main__':
    main(
        # ckpt_path='./checkpoints/7b-pretrain/7B-steps-4400-consolidated.pth',
        ckpt_path='./checkpoints/7b-finetune/7B-steps-6600-consolidated.pth',
        tokenizer_path='/home/michael/models/meta_llama2/tokenizer.model',
        imagebind_ckpt_path='/home/michael/models/ImageBind/imagebind_huge.pth',
    )
