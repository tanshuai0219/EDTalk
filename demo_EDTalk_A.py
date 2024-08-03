import os, sys
import torch
import torch.nn as nn
from networks.generator import Generator
from networks.audio_encoder import Audio2Lip
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from networks.utils import check_package_installed
from moviepy.editor import *


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    
    resized_frames = torch.stack([transform(frame) for frame in vid_norm[0]], dim=0).unsqueeze(0)
    return resized_frames, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)

import audio
def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def get_mel(audio_path):

    wav = audio.load_wav(audio_path, 16000) 
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()         # nframes 80
    indiv_mels = []
    fps = 25
    syncnet_mel_step_size = 16


    for i in range(num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).cuda()
    source_audio_feature = indiv_mels.type(torch.FloatTensor).cuda()

    mel_input = source_audio_feature                       # bs T 1 80 16
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

    return audiox, bs, T


def audio_preprocessing(wav_path):
    source_audio_feature, bs, T = get_mel(wav_path)

    return source_audio_feature, bs, T

class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args
        model_path = args.model_path
        audio2lip_model_path = args.audio2lip_model_path
        print('==> loading model')
        self.audio2lip = Audio2Lip().cuda()
        weight = torch.load(audio2lip_model_path, map_location=lambda storage, loc: storage)['audio2lip']
        self.audio2lip.load_state_dict(weight)
        self.audio2lip.eval()
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.latent_dim_exp, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        print('==> loading data')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()
        self.audio, self.bs, self.T = audio_preprocessing(args.audio_driving_path)

        if args.audio_driving_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Warning: The provided audio_driving_path is in video format. Please provide an audio file.")
           
        self.audio_path = args.audio_driving_path
        self.exp_vid_target, self.fps = vid_preprocessing(args.exp_driving_path)
        self.exp_vid_target = self.exp_vid_target.cuda()
        self.save_path = args.save_path
        self.pose_vid_target, self.fps = vid_preprocessing(args.pose_driving_path)
        self.pose_vid_target = self.pose_vid_target.cuda()
    def run(self):

        print('==> running')
        with torch.no_grad():
            # self.save_path = args.save_path
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            vid_target_recon = []

            h_start = None
            self.lip_vid_target = self.audio2lip(self.audio, self.bs, self.T)[0]
            self.lip_vid_target = conv_feat(self.lip_vid_target, k_size=3, sigma=1) # torch.Size([372, 500])

            self.exp_vid_target = self.exp_vid_target[:,:-20]

            while  self.exp_vid_target.shape[1]<self.lip_vid_target.size(0):
                self.exp_vid_target = torch.cat([self.exp_vid_target, torch.flip(self.exp_vid_target, dims =[1])], dim=1)
            self.exp_vid_target = self.exp_vid_target[:self.lip_vid_target.size(0)]

            exp_len = self.exp_vid_target.shape[1]
            len_pose = self.pose_vid_target.shape[1]

            for i in tqdm(range(self.lip_vid_target.size(0))):
                img_target_lip = self.lip_vid_target[i:i+1]
                if i>=len_pose:
                    img_target_pose = self.pose_vid_target[:, -1, :, :, :]
                else:
                    img_target_pose = self.pose_vid_target[:, i, :, :, :]
                if i>=exp_len:
                    img_target_exp = self.exp_vid_target[:, -1, :, :, :]
                else:
                    img_target_exp = self.exp_vid_target[:, i, :, :, :]
                img_recon = self.gen.test_EDTalk_A(self.img_source, img_target_lip, img_target_pose, img_target_exp, h_start)
                
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            
            temp_path = self.save_path.replace('.mp4','_temp.mp4')
            save_video(vid_target_recon, temp_path, self.fps)
            cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_path, self.audio_path, self.save_path)
            os.system(cmd)
            os.remove(temp_path)
            
            if args.face_sr and check_package_installed('gfpgan'):
                from face_sr.face_enhancer import enhancer_list
                import imageio

                temp_512_path = self.save_path.replace('.mp4','_512.mp4')

                # Super-resolution
                imageio.mimsave(temp_512_path + '.tmp.mp4', enhancer_list(self.save_path, method='gfpgan', bg_upsampler=None), fps=float(25), codec='libx264')
                
                # Merge audio and video
                video_clip = VideoFileClip(temp_512_path + '.tmp.mp4')
                audio_clip = AudioFileClip(self.save_path)
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(temp_512_path, codec='libx264', audio_codec='aac')
                
                os.remove(temp_512_path + '.tmp.mp4')

def conv_feat(features, k_size, weight=None, sigma=1.0):
    c = features.shape[1] # torch.Size([101, 500])
    if weight is None:
        pad = k_size // 2
        k = np.zeros(k_size).astype(np.float)
        for x in range(-pad, k_size-pad):
            k[x+pad] = np.exp(-x**2 / (2 * (sigma ** 2)))
        k = k / k.sum()
        print(k) # [0.27406862 0.45186276 0.27406862]
    else:
        k_size = len(weight)
        k = np.array(weight)
        pad = k_size // 2
        print(k)
    
    k = torch.from_numpy(k).to(features.device).float().unsqueeze(0).unsqueeze(0)
    k = k.repeat(c, 1, 1)
    features = features.unsqueeze(0).permute(0, 2, 1) # [1, 512, n]
    features = F.conv1d(features, k, padding=pad, groups=c)
    features = features.permute(0, 2, 1).squeeze(0)

    return features

if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_exp", type=int, default=10)
    parser.add_argument("--source_path", type=str, default='test_data/identity_source.jpg')
    parser.add_argument("--audio_driving_path", type=str, default='test_data/mouth_source.wav')
    parser.add_argument("--pose_driving_path", type=str, default='test_data/pose_source1.mp4')
    parser.add_argument("--exp_driving_path", type=str, default='test_data/expression_source.mp4')
    parser.add_argument("--save_path", type=str, default='res/demo_EDTalk_A.mp4')
    parser.add_argument("--audio2lip_model_path", type=str, default='ckpts/Audio2Lip.pt')
    parser.add_argument("--model_path", type=str, default='ckpts/EDTalk.pt')
    parser.add_argument('--face_sr', action='store_true', help='Face super-resolution (Optional). Please install GFPGAN first')

    args = parser.parse_args()

    demo = Demo(args)
    demo.run()


