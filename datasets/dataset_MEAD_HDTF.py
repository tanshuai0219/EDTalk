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
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')



class VoxDataset(Dataset):
    def __init__(self, opt, is_inference, transform = None):
        path = opt.path
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "lists/MEAD_HDTF_test.json" if is_inference else "lists/MEAD_HDTF_train.json"
        # list_file = os.path.join(path, "lists",list_file)

        with open(list_file,"r") as f:
            videos = json.load(f)
        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids, self.person_id_meads, self.person_ids_emotion = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.idx_by_person_id_emotion = self.group_by_key(self.video_items, key='person_id_emotion')
        self.idx_by_person_id_mead = self.group_by_key(self.video_items, key='person_id_mead')
        self.person_ids = self.person_ids #* 100
        self.transform = transform

    def get_video_index(self, videos):
        video_items = []
        person_ids = []
        person_id_meads = []
        person_ids_emotion = []
        tot_len = len(videos)
        print('loading video_index')
        pbar = tqdm(range(tot_len))
        for i in pbar:
            video  = videos[i]
            video_items.append(self.Video_Item(video))
            splits = video.split('#')
            if len(splits) == 2:
                person_ids.append(splits[0])
                person_id_meads.append('lll')
                person_ids_emotion.append(splits[0]+'#'+splits[1])
            else:
                a,b,c,d = splits
                person_ids.append( a+'#'+b+'#'+c)
                person_id_meads.append(a)
                person_ids_emotion.append(splits[0]+'#'+splits[1])

        person_ids = sorted(person_ids)
        person_id_meads = sorted(person_id_meads)
        person_ids_emotion = sorted(person_ids_emotion)

        return video_items, person_ids, person_id_meads, person_ids_emotion

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        splits = video_name.split('#')
        if len(splits) == 2:
            video_item['video_name'] = video_name
            video_item['person_id'] = video_name.split('#')[0] # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = 'lll'
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item
        else:
            a,b,c,d = splits
            video_item['video_name'] = video_name
            video_item['person_id'] = a+'#'+b+'#'+c # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = a
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        
        if len(person_id.split('#'))==3:
            person_id_neutral = person_id.split('#')[0]+'#neutral'
            source_video_item = self.video_items[random.choices(self.idx_by_person_id_emotion[person_id_neutral], k=1)[0]]
            
            target_video_item = self.video_items[random.choices(self.idx_by_person_id_mead[person_id.split('#')[0]], k=1)[0]]
            frame_source, frame_target = self.random_select_frames_source_target(source_video_item,target_video_item)
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(source_video_item['video_name'], frame_source) # M027#neutral#014-0000153
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(target_video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)
                # print(video_item['video_name'], frame_source, frame_target)



            img1 = Image.open(BytesIO(img_bytes_1)) 

            img2 = Image.open(BytesIO(img_bytes_2))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            data['source_image'] = img1
            data['target_image'] = img2

            data['source_name'] = source_video_item['video_name']+'#'+str(frame_source)
            data['driving_name'] = target_video_item['video_name']+'#'+str(frame_target)
        
            return data
        else:
            video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
            
            frame_source, frame_target = self.random_select_frames(video_item)

            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], frame_source)
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)

            img1 = Image.open(BytesIO(img_bytes_1)) 

            img2 = Image.open(BytesIO(img_bytes_2))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            data['source_image'] = img1
            data['target_image'] = img2

            
            data['source_name'] = video_item['video_name']+'#'+str(frame_source)
            data['driving_name'] = video_item['video_name']+'#'+str(frame_target)

            return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def random_select_frames_source_target(self, source_video_item, target_video_item):
        num_frame = source_video_item['num_frame']
        from_frame_idx = random.choices(list(range(num_frame)), k=1)
        num_frame_target = target_video_item['num_frame']
        target_frame_idx = random.choices(list(range(num_frame_target)), k=1)
        return from_frame_idx[0], target_frame_idx[0]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq



