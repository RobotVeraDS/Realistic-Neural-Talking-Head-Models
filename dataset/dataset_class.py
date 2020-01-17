import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

from .video_extraction_conversion import *


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device, join_by_video=False):
        self.K = K
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.join_by_video = join_by_video

        self.idx_to_info = []

        # init
        idx = 0
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):

                if self.join_by_video:
                    self.idx_to_info.append((person_id, video_id))
                    idx += 1
                else:
                    for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                        self.idx_to_info.append((person_id, video_id, video))
                        idx += 1

    def __len__(self):
        return len(self.idx_to_info)

    def get_video_info(self, idx):
        if idx < 0:
            idx = self.__len__() + idx

        return self.idx_to_info[idx]

    def get_frame_mark_numpy_array(self, idx):
        path = os.path.join(self.path_to_mp4, *self.get_video_info(idx))

        frame_mark = select_frames(path, self.K, join_by_video=self.join_by_video)
        frame_mark = generate_landmarks(frame_mark, self.device)

        frame_mark = np.array(frame_mark)
        return frame_mark

    def __getitem__(self, idx):
        frame_mark = self.get_frame_mark_numpy_array(idx)

        frame_mark = torch.from_numpy(frame_mark).type(dtype = torch.float)  # K,2,224,224,3
        frame_mark = frame_mark.transpose(2, 4).to(self.device)  # K,2,3,224,224

        g_idx = torch.randint(low = 0, high = self.K, size = (1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, idx


class PreprocessedVidDataSet(Dataset):

    def __init__(self, K, path_to_data, device):
        self.K = K
        self.path_to_data = path_to_data
        self.device = device
        self.filenames = os.listdir(self.path_to_data)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        is_ok = False
        while not is_ok:
            try:
                data_path = os.path.join(self.path_to_data,
                                         self.filenames[idx])
                frame_mark = np.load(data_path)["frame_mark"]

                if frame_mark.shape[0] < (self.K + 1):
                    print("Warning! Idx: {} has {} frames that less than {}".format(
                        idx, frame_mark.shape[0], self.K + 1))
                    idx = random.randint(0, len(self.filenames) - 1)
                    print("Generated new idx: {}".format(idx))
                    continue

                random_frames_idxs = np.random.choice(range(frame_mark.shape[0]),
                                                      self.K + 1, replace=False)
                frame_mark = frame_mark[random_frames_idxs, :, :, :]

                frame_mark = torch.from_numpy(frame_mark).type(dtype = torch.float) #  K+1,2,224,224,3
                frame_mark = frame_mark.transpose(2, 4).to(self.device) #  K+1,2,3,224,224
                is_ok = True

            except:
                print("Warning! {} failed".format(idx))
                idx = random.randint(0, len(self.filenames) - 1)
                print("Generated new idx: {}".format(idx))

        # get first image from random K + 1 as target
        x = frame_mark[0, 0].squeeze()
        g_y = frame_mark[0, 1].squeeze()

        frame_mark = frame_mark[1:]  # other K images for embedding

        return frame_mark, x, g_y, idx


class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device

    def __len__(self):
        return len(os.listdir(self.path_to_images))

    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low = 0, high = len(frame_mark_images), size = (1,1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(frame_mark_images, pad=50, device=self.device)
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2,4).to(self.device) #1,2,3,256,256

        x = frame_mark_images[0, 0].squeeze()
        g_y = frame_mark_images[0,1].squeeze()

        return x, g_y


class FineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device):
        self.path_to_video = path_to_video
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path , 1)
                frame_mark = generate_cropped_landmarks(frame_mark, pad=50, device=self.device)
                frame_has_face = True
            except:
                print('No face detected, retrying')
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(self.device) #1,2,3,256,256

        x = frame_mark[0,0].squeeze()
        g_y = frame_mark[0,1].squeeze()
        return x, g_y
