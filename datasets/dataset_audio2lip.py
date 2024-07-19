import os
from unittest import main
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import pickle
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch, json
from io import BytesIO
import cv2
import torch.nn.functional as F
from torchvision import utils
import numpy as np
from PIL import Image
import cv2
import skimage.transform as trans
import lmdb
import torchvision

    

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



class Audio2LipDataset_image_sync(Dataset):
    def __init__(self, hdtf, is_train=True, transform = None):
        self.is_train = is_train
        self.hdtf_path = hdtf
        self.env = lmdb.open(
            'EDTalk_lmdb',
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        # self.augmenter = augmentation.ParametricAugmenter()
        
        if self.is_train:
            self.type = 'train'
            file = 'lists/HDTF_train_bbox.json'

            with open(file,"r") as f:
                self.video_list = json.load(f)
        
        else:
            self.type = 'test'
            file = 'lists/HDTF_test_bbox.json'
            with open(file,"r") as f:
                self.video_list = json.load(f)

        random.shuffle(self.video_list)

        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        idx_a = idx
        name_a = self.video_list[idx_a]
        

        audio_path = os.path.join(self.hdtf_path,'mel', name_a+'.npy')
        audio_features = np.load(audio_path)

        lip_path = os.path.join(self.hdtf_path,'EDTalk_lip_feature', name_a+'.npy')
        lip_features = np.load(lip_path)

        pose_path = os.path.join(self.hdtf_path,'EDTalk_pose_feature', name_a+'.npy')
        pose_features = np.load(pose_path)

        bbox_path = os.path.join(self.hdtf_path,'bbox', name_a+'.npy')
        bbox_features = np.load(bbox_path)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(name_a, 'length')
            length = int(txn.get(key).decode('utf-8'))
        l = min(min(len(audio_features), len(lip_features)),length)
        r = random.choice([x for x in range(l-6)])
        r_identity = random.choice([x for x in range(l-1)])
        


        image_list = []

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(name_a, r_identity)
            img_bytes = txn.get(key)
            identity_img = self.transform(Image.open(BytesIO(img_bytes)))
            for current_frame in range(r,r+5):
                key = format_for_lmdb(name_a, current_frame)
                img_bytes = txn.get(key)
                image_list.append(self.transform(Image.open(BytesIO(img_bytes))))

        image_list = torch.stack(image_list, dim=0)

        data = {}
        data['audio_features'] = audio_features[r:r+5]
        data['lip_features'] = lip_features[r:r+5]
        data['pose_features'] = pose_features[r:r+5]
        data['identity_img'] = identity_img
        data['target_img'] = image_list

        bbox_len = len(bbox_features)
        if r+5<bbox_len:
            data['bbox'] = bbox_features[r:r+5]
        else:
            bbox = []

            for i in range(r,r+5):
                try:
                    temp = bbox_features[i]
                except:
                    temp = bbox_features[-1]
                bbox.append(temp)
            bbox = np.array(bbox)
            data['bbox'] = bbox
        return data
