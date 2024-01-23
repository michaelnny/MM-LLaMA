# MM-LLaMA

Bring Multimodal to the LLaMA model by leverage ImageBind as the vision encoder. Supports vision (image and short video) input to the LLaMA model, and text output from LLaMA.

# Disclaimer

**Project Purpose:** This project is for research and education only, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** We run some testing upon working on the project, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

**Optimization:** For simplicity, we only focus on training on a single GPU, as the PyTorch FSDP and QLoRA seems not working very well yet. Additionally, the hyper-parameters for the different training scripts are not fine-tuned.

**Model Performance:** The final performance of the fine-tuned model is acceptable but not excellent (~70% accuracy). Is can identify and correctly classify the objects in images/videos. However, when asking to describe the details, sometimes it generates things that not present in the given media. In this case, more training data could be beneficial.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.1.1
- Tensorboard 2.13.0
- Bitsandbytes 0.41.3

# Code Structure

- `mm_llama` directory contains main source code for the project.
  - `configs` directory contains all the training configurations like model type, data source, number of iterations, learning rate etc.
  - `utils` directory contains helper modules like custom datasets, logging, tokenization, LoRA module etc.
  - `models` contains the LLaMA model class and ImageBind model and it's data processing utilities.
  - `run_train.py` run pre-training or fine-tuning, using LoRA parameter efficient fine-tuning method (only supports single GPU).
  - `chat_completion.py` run evaluation chat completion with the trained model.
- `scripts` directory contains all source code for convert the model weights and build datasets for different phases.
  - `build_datasets.py` build pre-train, fine-tuning datasets (save the dataset to .pkl files).
  - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `convert_lora_checkpoint.py` convert fine-tunned LoRA weights to a full state_dict checkpoint.
- `logs` directory contains training logs for the different phases.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```

# Project Preparation

_Notice: The scripts in the project uses hard-coded file paths which may not exists in your environment. You should change these to suit your environment before you run any script_

## Download and prepare LLaMA chat model weights

1. **Download the fine-tuned chat model weights** please refer to https://github.com/facebookresearch/llama on how to download it.
2. **Convert Meta's fine-tuned chat model weights** using script `python3 scripts/convert_meta_checkpoint.py`, so it's compatible with our naming convention.

## Download ImageBind model checkpoint

The following script to download the ImageBind (huge) model checkpoint. By default, it will download the save the checkpoint file to './checkpoints/imagebind/imagebind_huge.pth'

```
python3 scripts/download_imagebind_checkpoint.py
```

## Download datasets

We use three datasets for the project:

- LLaVA instruct chat dataset based on CC-3M image dataset for stage 1 pre-training the modality-to-LLM alignment projection layer
- LLaVA instruct 150k instruct dataset based on COCO image dataset for stage 2 fine-tuning the model
- VideoChat instruct 11k dataset based on WebVid-10M video dataset for stage 2 fine-tuning the model

To prepare each of the above mentioned dataset, we need to first download the raw images/videos, them download the instruct .json files.

We want to thank the authors of [LLaVA project](https://github.com/haotian-liu/LLaVA) and [InternVideo project](https://github.com/OpenGVLab/InternVideo) for publishing their datasets. More details about the datasets can be found here:

- https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md
- https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data

### 1. Prepare LLaVA instruct chat CC-3M image datasets

**Download image files**
Download the CC-3M raw image dataset from the link:
https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip

**Download instruct/chat files**

Download the instruct chat.json file for CC-3M dataset from the link:
https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json

### 2. Prepare LLaVA instruct 150k instruct COCO image datasets

**Download image files**

Download the COCO image dataset from the link: http://images.cocodataset.org/zips/train2017.zip

**Download instruct/chat files**

Download the instruct chat.json file for COCO dataset from the link:
https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json

### 3. Prepare VideoChat instruct 11k video datasets

The original WebVid-10M dataset is very large (~2TB), since the VideoChat only uses ~8k videos from the WebVid-10M dataset, we've come up with a script to only download these ~8k videos.

To do so, we first need to download the metadata for the WebVid-10M (train)dataset from https://maxbain.com/webvid-dataset/

We then need to download the VideoChat instruction .json file from https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data

After these two files (metadata .csv and instruction .json) files are ready, we can run the following script to download the ~8k videos.

```
python3 scripts/download_webvid_dataset.py
```

## Build datasets

To build the training datasets, we need the ImageBind model checkpoint in order to pre-compute the embeddings or hidden features for the images/videos. We chose to pre-compute the embeddings for the images/videos for the following reasons:

- Simplicity: doing so keep the changes to the LLaMA model minimum
- Save compute: we can batch processing the datasets once, and the using the dataset multiple times or epochs
- Save GPU memory during training: once the embeddings are computed, the ImageBind model is not needed during the training phase, thus saves GPU memory

Once the dataset raw images/videos files are ready, and the instruct .json files also been prepared. We can start build the datasets by running the following script.

```
python3 scripts/build_datasets.py
```

We following the same prompt template from Meta's LLaMA 2 model, however we add modality specific tokens `<<IMG>>`, `<</IMG>>`, `<<VID>>`, `<</VID>>` to the prompt. For example, for image input, the prompt template looks like this:

```
<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} <<IMG>>UNK<</IMG>> [/INST] {1st response} </s>
```

where `UNK` is the unknown token.

# Training Stages

## Stage 1: MM-to-LLM alignment projection pre-training

Run the `python3 mm_llama/run_train.py --stage=1` script to train the modality-to-LLM alignment projection layer. Where we load the fine-tuned chat model from Meta's weights, where the LLM model is frozen. We only use the CC-3M (~500k) dataset.

Note after training is finished, we need to run the `python3 scripts/convert_lora_checkpoint.py` to convert the checkpoints for stage 2, although we don't use LoRA in stage 1, but we need to merge the LLM alignment weights.

## Stage 2: MM-to-LLM jointed fine-tuning with LoRA

Run the `python3 mm_llama/run_train.py --stage=2` script to jointly fine-tune the modality-to-LLM alignment projection layer and the LLM model. We use LoRA to fine-tune the LLM and full fine-tune (no LoRA) the modality-to-LLM alignment projection layer. We use the LLaVA instruct (150k) and Videochat instruct (11k) datasets.

Note after training is finished, we need to run the `python3 scripts/convert_lora_checkpoint.py` to convert the checkpoints to merge LoRA weights.

# Monitoring with Tensorboard

We can monitoring the training progress by using Tensorboard:

```
tensorboard --logdir=./logs
```

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

- The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license

- The ImageBind code and model weights are released under the CC-BY-NC 4.0 license. For details, visit: https://github.com/facebookresearch/ImageBind#license

# Acknowledgments

This project is greatly inspired by the following projects:

- [Llama 2] (https://github.com/facebookresearch/llama)
- [ImageBind] (https://github.com/facebookresearch/ImageBind)
- [MiniGPT-4] (https://github.com/Vision-CAIR/MiniGPT-4)
- [LoRA] (https://github.com/microsoft/LoRA)
- [InstructLLaMA] (https://github.com/michaelnny/InstructLLaMA)
