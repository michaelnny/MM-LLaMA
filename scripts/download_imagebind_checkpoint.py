import os
import argparse
import torch


CKPT_URL = 'https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth'


def download_file(save_path: str):
    # Check if the file already exists
    if os.path.exists(save_path):
        print('File already exists. Aborting...')
        return

    # Try to create folder
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Download the file using urllib
    try:
        torch.hub.download_url_to_file(
            CKPT_URL,
            save_path,
            progress=True,
        )
        print(f'File downloaded successfully. Saved to: {save_path!r}')
    except Exception as e:
        print(f'Failed to download the file. Error: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ImageBind huge checkpoint')
    parser.add_argument(
        '--save_path',
        help='Full path to save the ImageBind checkpoint',
        type=str,
        default='./checkpoints/imagebind/imagebind_huge.pth',
        nargs='?',
    )
    args = parser.parse_args()

    download_file(args.save_path)
