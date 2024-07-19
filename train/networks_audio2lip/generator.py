from torch import nn
from .encoder import *
from .styledecoder import Synthesis
import torch

class Direction(nn.Module):
    def __init__(self, lip_dim, pose_dim):
        super(Direction, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim+pose_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out
    def get_shared_out(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)  # torch.Size([1, 20, 512])
            return out
            # out = torch.sum(out, dim=1)

            # return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent

class Generator(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)
        # self.source_fc = EqualLinear(style_dim, motion_dim)

        lip_fc = [EqualLinear(style_dim, style_dim)]
        lip_fc.append(EqualLinear(style_dim, style_dim))
        lip_fc.append(EqualLinear(style_dim, lip_dim))
        self.lip_fc = nn.Sequential(*lip_fc)

        pose_fc = [EqualLinear(style_dim, style_dim)]
        pose_fc.append(EqualLinear(style_dim, style_dim))
        pose_fc.append(EqualLinear(style_dim, pose_dim))
        self.pose_fc = nn.Sequential(*pose_fc)


    def forward(self, img_source, img_drive, h_start=None):
        wa, wa_t, feats, feats_t = self.enc(img_source, img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)
        alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_lip_nonlip(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon
    
    def get_lip_pose_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(img_source) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        alpha_D_lip = self.lip_fc(shared_fc_p)

        return alpha_D_lip, alpha_D_pose