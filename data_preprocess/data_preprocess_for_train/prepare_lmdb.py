import os
import cv2
import lmdb
import argparse
import multiprocessing
import numpy as np

from glob import glob
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat
from torchvision.transforms import functional as trans_fn

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



class Resizer_MEAD_HDTF:
    def __init__(self, MEAD_path, HDTF_path):
        self.MEAD_path = MEAD_path
        self.HDTF_path = HDTF_path
        self.img_format = 'jpeg'
        self.size = 256

    def prepare_HDTF(self, filename):
        # print(filename)
        frames = {'img':[]}
        video_name = os.path.join(self.HDTF_path, 'split_5s_video', filename+'.mp4')

        cap = cv2.VideoCapture(video_name)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                img_bytes = self.get_resized_bytes(img_pil, self.img_format)
                frames['img'].append(img_bytes)
            else:
                break
        cap.release()

        return frames

    def prepare_MEAD(self, filename):
        # print(filename)
        frames = {'img':[]}
        video_name = os.path.join(self.MEAD_path, 'video', filename+'.mp4')

        cap = cv2.VideoCapture(video_name)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame)
                img_bytes = self.get_resized_bytes(img_pil, self.img_format)
                frames['img'].append(img_bytes)
            else:
                break
        cap.release()

        return frames

    def get_resized_bytes(self, img, img_format='jpeg'):
        img = trans_fn.resize(img, (self.size, self.size), interpolation=Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format=img_format)
        img_bytes = buf.getvalue()
        return img_bytes

    def __call__(self, index_filename):
        index, filename = index_filename
        if len(filename.split('#')) == 2:
            result = self.prepare_HDTF(filename)
        else:
            result = self.prepare_MEAD(filename)
        return index, result, filename



import json
def prepare_data(MEAD_path,HDTF_path, out, n_worker, chunksize):
    
    train_file = 'data_preprocess/lists/train.json'
    with open(train_file,"r") as f:
        train_video = json.load(f)
    test_file = 'data_preprocess/lists/test.json'
    with open(test_file,"r") as f:
        test_video = json.load(f)
    
    filenames = train_video+test_video

    filenames = sorted(filenames)
    total = len(filenames)
    os.makedirs(out, exist_ok=True)
    lmdb_path = out
    with lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            txn.put(format_for_lmdb('length'), format_for_lmdb(total))
            resizer = Resizer_MEAD_HDTF(MEAD_path,HDTF_path)
            with multiprocessing.Pool(n_worker) as pool:
                for idx, result, filename in tqdm(
                        pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                        total=total):
                    filename = os.path.basename(filename)
                    video_name = os.path.splitext(filename)[0]
                    # print(video_name)
                    txn.put(format_for_lmdb(video_name, 'length'), format_for_lmdb(len(result['img'])))

                    for frame_idx, frame in enumerate(result['img']):
                        txn.put(format_for_lmdb(video_name, frame_idx), frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MEAD_path', type=str, help='a path to input directiory', default='MEAD_front')
    parser.add_argument('--HDTF_path', type=str, help='a path to input directiory', default='HDTF')
    parser.add_argument('--out', type=str, help='a path to output directory', default = 'EDTalk_lmdb')
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    args = parser.parse_args()
    prepare_data(**vars(args))