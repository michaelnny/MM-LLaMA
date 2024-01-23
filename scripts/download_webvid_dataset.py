import os
from typing import Tuple
import tqdm
import json
import argparse
import requests
import pandas as pd
import multiprocessing as mp


def download_single_video(args: Tuple[str]) -> None:
    url, save_path = args

    if not url or url == '':
        return
    if os.path.exists(save_path):
        return

    try:
        video_data = requests.get(url, timeout=5).content
        with open(save_path, 'wb') as handler:
            handler.write(video_data)
    except Exception:
        pass


def main(args):
    assert args.num_workers >= 1

    if not os.path.exists(args.meta_csv_file):
        print(f'Invalid csv meta file {args.meta_csv_file!r}. Aborting...')
        return
    if os.path.exists(args.save_dir) and len(os.listdir(args.save_dir)) > 0:
        print(f'Directory {args.save_dir!r} is not empty. Aborting...')
        return

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Get video ids from the chat json file
    print('Reading chat json file, this may take a while...')
    chat_videoids = []
    with open(args.chat_json_file, encoding='utf-8') as f:
        content = f.read()
        content = json.loads(content)

        for item in content:
            try:
                # "050951_051000/1022713978.mp4" -> "1022713978"
                videoid = item['video'].split('/')[1].split('.mp4')[0]
                chat_videoids.append(str(videoid))
            except Exception:
                pass

    # Get video download urls
    print('Reading metadata csv file, this may take a while...')
    meta_df = pd.read_csv(args.meta_csv_file, usecols=['videoid', 'contentUrl'])
    meta_df['videoid'] = meta_df['videoid'].astype(str)
    meta_df['save_path'] = meta_df.apply(lambda x: os.path.join(args.save_dir, x['videoid'] + '.mp4'), axis=1)

    # filter out videos not in the chat json file
    df = meta_df[meta_df['videoid'].isin(chat_videoids)]

    download_list = list(zip(df['contentUrl'], df['save_path']))

    print(f'Starting to download {len(download_list)} videos...')
    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(download_single_video, download_list), total=len(download_list), desc='Downloading'))


if __name__ == '__main__':
    # Set multiprocessing start mode
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Download VideoChat dataset video files (8k) from the original 10M WebVid dataset')
    parser.add_argument(
        '--meta_csv_file',
        help='Path to the WebVid metadata csv file',
        type=str,
        default='/home/michael/datasets/WebVid/results_10M_train.csv',
    )
    parser.add_argument(
        '--chat_json_file',
        help='Path to VideoChat instruction json file',
        type=str,
        default='/home/michael/datasets/WebVid/videochat_instruct_11k.json',
    )
    parser.add_argument(
        '--save_dir',
        help='Path to save the downloaded video files',
        type=str,
        default='/home/michael/datasets/WebVid/videos',
    )
    parser.add_argument(
        '--num_workers',
        help='Number of parallel workers to download the videos',
        type=int,
        default=8,
    )
    args = parser.parse_args()

    main(args)
