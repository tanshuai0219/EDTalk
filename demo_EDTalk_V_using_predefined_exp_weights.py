import os, sys
import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

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


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        model_path = args.model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.latent_dim_exp, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()
        print('==> loading data')
        self.img_source = img_preprocessing(args.source_path, args.size).cuda()

        self.pose_vid_target, self.fps = vid_preprocessing(args.pose_driving_path)
        
        self.pose_vid_target = self.pose_vid_target.cuda()
        self.audio_path = args.audio_driving_path
        self.exp_vid_target = np.load(os.path.join('ckpts/predefined_exp_weights', args.exp_type+'.npy'))
        self.exp_vid_target = torch.from_numpy(self.exp_vid_target).cuda()

        self.save_path = args.save_path
        self.lip_vid_target, self.fps = vid_preprocessing(args.lip_driving_path)
        self.lip_vid_target = self.lip_vid_target.cuda()
    def run(self):

        print('==> running')
        with torch.no_grad():
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            vid_target_recon = []

            h_start = None

            pose_len = self.pose_vid_target.shape[1]
            for i in tqdm(range(self.lip_vid_target.size(1))):

                img_target_lip = self.lip_vid_target[:, i, :, :, :]
                if i>=pose_len:
                    img_target_pose = self.pose_vid_target[:, -1, :, :, :]
                else:
                    img_target_pose = self.pose_vid_target[:, i, :, :, :]

                img_recon = self.gen.test_EDTalk_V_use_exp_weight(self.img_source, img_target_lip, img_target_pose, self.exp_vid_target, h_start)

                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            
            temp_path = self.save_path.replace('.mp4','_temp.mp4')
            save_video(vid_target_recon, temp_path, self.fps)
            cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_path, self.audio_path, self.save_path)
            os.system(cmd)
            os.remove(temp_path)


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_exp", type=int, default=10)
    parser.add_argument("--source_path", type=str, default='demo/identity_source.jpg')
    parser.add_argument("--lip_driving_path", type=str, default='demo/mouth_source.mp4')
    parser.add_argument("--audio_driving_path", type=str, default='demo/mouth_source.wav')
    parser.add_argument("--pose_driving_path", type=str, default='demo/pose_source1.mp4')
    parser.add_argument("--exp_type", type=str, default='contempt') # ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised']
    parser.add_argument("--save_path", type=str, default='res/demo_EDTalk_V_using_weights.mp4')
    parser.add_argument("--model_path", type=str, default='ckpts/EDTalk.pt')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run()

