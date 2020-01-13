import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

from .video_extraction_conversion import *

def create_filename(person_id, video_id, video):
    filename = "{}_{}_{}.torch".format(person_id, video_id, video)
    return filename


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device):
        self.K = K
        self.path_to_mp4 = path_to_mp4
        self.device = device

        self.idx_to_info = []

        # init
        idx = 0
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
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
        person_id, video_id, video = self.get_video_info(idx)

        path = os.path.join(self.path_to_mp4, person_id, video_id, video)

        frame_mark = select_frames(path, self.K)
        frame_mark = generate_landmarks(frame_mark, self.device)

        return frame_mark

    def __getitem__(self, idx):
        vid_idx = idx
        person_id, video_id, video = self.get_video_info(idx)

        path = os.path.join(self.path_to_mp4, person_id, video_id, video)

        frame_mark = select_frames(path, self.K)
        frame_mark = generate_landmarks(frame_mark, self.device)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #  K,2,224,224,3
        frame_mark = frame_mark.transpose(2, 4).to(self.device) #  K,2,3,224,224

        g_idx = torch.randint(low = 0, high = self.K, size = (1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

        return frame_mark, x, g_y, vid_idx


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
                frame_mark = torch.load(data_path)

                frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #  K,2,224,224,3
                frame_mark = frame_mark.transpose(2, 4).to(self.device) #  K,2,3,224,224
                is_ok = True
            except:
                print("Warning. {} failed".format(idx))
                idx = random.randint(0, len(self.filenames) - 1)
                print("New idx: {}".format(idx))

        g_idx = torch.randint(low = 0, high = self.K, size = (1, 1))
        x = frame_mark[g_idx, 0].squeeze()
        g_y = frame_mark[g_idx, 1].squeeze()

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
