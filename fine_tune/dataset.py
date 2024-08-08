import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Finetune256(Dataset):
    def __init__(self, video_path, train, transform=None):
        
        train_test_split = 0.8
        self.train = train
       
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        video_len = int(len(frames_paths)*train_test_split)
        
        if self.train:
            self.frames_paths = frames_paths[:video_len]
        else:
            self.frames_paths = frames_paths[video_len:]
        self.transform = transform

    def __getitem__(self, idx):
        frames_paths = self.frames_paths
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')


        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.frames_paths)

