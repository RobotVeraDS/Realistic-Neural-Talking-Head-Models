from dataset.dataset_class import VidDataSet
import torch
import tqdm
import os
import argparse
import random
import numpy as np


def create_filename(*args):
    filename = "{}.npz".format("_".join(args))
    return filename


def main(args):
    dataset = VidDataSet(K=args.K,
                         path_to_mp4=args.data_dir,
                         device=torch.device("cuda"),
                         join_by_video=True)
    print("Dataset size", len(dataset))

    for idx in tqdm.tqdm(range(args.start_idx, args.end_idx + 1)):
        filename = create_filename(*dataset.get_video_info(idx))
        output_path = os.path.join(args.output_dir, filename)

        frame_mark = dataset.get_frame_mark_numpy_array(idx)
        np.savez_compressed(output_path, frame_mark=frame_mark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess videos for training.')

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, required=True)

    parser.add_argument("--start_idx", type=int, required=True, default=0)
    parser.add_argument("--end_idx", type=int, required=True, default=1)

    args = parser.parse_args()
    main(args)
