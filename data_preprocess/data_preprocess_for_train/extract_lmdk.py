import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import cv2
import dlib
from imutils import face_utils
import warnings
warnings.filterwarnings("ignore")

hdtf_save_dir = 'HDTF/landmark'
mead_save_dir = 'MEAD_front/landmark'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

unextract = []

env = lmdb.open(
    'EDTalk_lmdb',
    max_readers=32,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

if not env:
    raise IOError('Cannot open lmdb dataset', 'EDTalk_lmdb')



def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



def extract(video):
    with env.begin(write=False) as txn:
        key = format_for_lmdb(video, 'length')
        length = int(txn.get(key).decode('utf-8'))
        landmark = []
        for j in range(length):
            key = format_for_lmdb(video, j) # M027#neutral#014-0000153
            img_bytes = txn.get(key)
            img = Image.open(BytesIO(img_bytes))
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # 检测人脸并检测landmarks
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                landmark.append(shape)
        landmark = np.array(landmark)
        splits = video.split('#')
        if len(splits)==2:
            np.save(os.path.join(hdtf_save_dir, video+'.npy'),landmark)
        else:
            np.save(os.path.join(mead_save_dir, video+'.npy'),landmark)
  
        print(video)


from multiprocessing import Pool
import multiprocessing as mp

import math, time
from argparse import ArgumentParser



def multi_pro(args):
    for arg in args:
        try:
            extract(arg)
        except:
            unextract.append(arg)
            print('skip!')

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]
                
if __name__ == "__main__":
    parser = ArgumentParser()
    mp.set_start_method('spawn')

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--inp", default=None, help='Input image or video')
    parser.add_argument("--outp", default=None, help='Input image or video')
    parser.add_argument("--min_frames", type=int, default=0,  help='Minimum number of frames')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--num_processes", type=int, default=10, help="Number of processes to use")
    parser.add_argument("--outp_audio", type=int, default=8, help="Number of processes to use")
    
    args = parser.parse_args()



    videos = []
    with open('data_preprocess/lists/train.json',"r") as f:
        videos += json.load(f)
    with open('data_preprocess/lists/test.json',"r") as f:
        videos += json.load(f)
    print(len(videos))
    file_chunks = list(chunks(videos, int(math.ceil(len(videos) / args.num_processes))))
    print(len(file_chunks))
    pool = mp.Pool(args.num_processes)
    tic = time.time()
    pool.map(multi_pro, file_chunks)
    toc = time.time()
    print('Mischief managed in {}s'.format(toc - tic))


