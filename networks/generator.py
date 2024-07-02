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

class Direction_exp(nn.Module):
    def __init__(self, lip_dim, pose_dim, exp_dim):
        super(Direction_exp, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.weight = nn.Parameter(torch.randn(512, exp_dim))

    def forward(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8 # torch.Size([512, 36])
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 36]) torch.Size([1, 36, 36])
            out = torch.matmul(input_diag, Q.T) # Q torch.Size([512, 36]) OUT torch.Size([1, 36, 512])
            out = torch.sum(out, dim=1)

            return out

    def only_exp(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 40, 40])
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

    def get_shared_out(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8
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
        pose_latent = torch.sum(out[:,self.lip_dim:self.lip_dim+self.pose_dim], dim=1)
        return pose_latent

    def get_exp_latent(self, out):
        exp_latent = torch.sum(out[:,self.lip_dim+self.pose_dim:], dim=1)
        return exp_latent


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        self.direction_exp = Direction_exp(lip_dim, pose_dim, exp_dim)
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

        exp_fc = [EqualLinear(style_dim, style_dim)]
        exp_fc.append(EqualLinear(style_dim, style_dim))
        exp_fc.append(EqualLinear(style_dim, exp_dim))
        self.exp_fc = nn.Sequential(*exp_fc)


    def test_EDTalk_V(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_V_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t, feats, _ = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_A(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon


    def test_EDTalk_A_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon