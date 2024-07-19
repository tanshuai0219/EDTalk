from torch import nn
from .encoder import *
from .styledecoder import Synthesis, Synthesis_with_EAM, Synthesis_with_EAM2, Synthesis_with_ADAIN, Synthesis_with_warp_ADAIN
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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon
    
    def test_only_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        e = self.direction_exp.get_exp_latent(a)
        latent_poseD = wa + e 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

        # a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        # e = self.direction_exp.get_exp_latent(a)

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def only_exp2(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)
        alpha_D = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1)
        directions_target_share = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_manipulate_exp(self, img_source,i,j):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa)
        alpha_D_exp = self.exp_fc(shared_fc_exp)
        alpha_D = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1)
        
        alpha_D[:,j] = alpha_D[:,j]*(1+0.1*i)
        directions_target_share = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_manipulate_exp2(self, img_source,i,j):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        alpha_D_exp[:,j] = alpha_D_exp[:,j]*(1+0.1*i)
        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats) 
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon


class Generator_nobank(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_nobank, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)
        # self.source_fc = EqualLinear(style_dim, motion_dim)

        lip_fc = [EqualLinear(style_dim, style_dim)]
        lip_fc.append(EqualLinear(style_dim, style_dim))
        lip_fc.append(EqualLinear(style_dim, 512))
        self.lip_fc = nn.Sequential(*lip_fc)

        pose_fc = [EqualLinear(style_dim, style_dim)]
        pose_fc.append(EqualLinear(style_dim, style_dim))
        pose_fc.append(EqualLinear(style_dim, 512))
        self.pose_fc = nn.Sequential(*pose_fc)

        exp_fc = [EqualLinear(style_dim, style_dim)]
        exp_fc.append(EqualLinear(style_dim, style_dim))
        exp_fc.append(EqualLinear(style_dim, 512))
        self.exp_fc = nn.Sequential(*exp_fc)

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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        # alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + alpha_D_pose+alpha_D_exp
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_only_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        # alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa +alpha_D_exp
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon


    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        # alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + alpha_D_lip+ alpha_D_pose+alpha_D_exp
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon


class Generator_using_EAM(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_using_EAM, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_with_EAM(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
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

    def test_evaluation_using_npy(self, wa, feats, alpha_D_lip,  alpha_D_pose, alpha_D_exp, h_start=None):

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def get_emotion_feature(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp) # torch.Size([72, 40])
        # alpha_D_exp = alpha_D_exp.mean(0).unsqueeze(0)
        # e = self.direction_exp.only_exp(alpha_D_exp) #torch.Size([1, 512])
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        alpha_target = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        directions_target_exp = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        return directions_target_exp

    def get_lip_pose_exp_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        alpha_D_lip = self.lip_fc(shared_fc_p)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        return alpha_D_lip, alpha_D_pose, alpha_D_exp

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_exp_from_pth(self, img_source, e, h_start=None):
        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)


        # alpha_target = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        # directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        # e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    
    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)


        alpha_target = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats, None)
        return img_recon

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def img_smooth(self, wa, feats, directions_D, e, h_start=None):
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def get_feat(self, img_source):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        return wa, feats

    def get_audio_feature(self, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa_t_p, wa_t_exp, _, _ = self.enc(pose_img_drive, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
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
        # latent_poseD = wa + directions_D 
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return directions_D, e

    def get_exp_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        return alpha_D_exp

class Generator_using_ADAIN(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_using_ADAIN, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_with_ADAIN(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_exp_0(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)
        alpha_D_exp = alpha_D_exp.mean(0)
        # e = self.direction_exp.only_exp(alpha_D_exp)
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return alpha_D_exp

    def only_exp_3_0(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp) # torch.Size([72, 40])
        alpha_D_exp = alpha_D_exp.mean(0).unsqueeze(0)
        # e = self.direction_exp.only_exp(alpha_D_exp) #torch.Size([1, 512])
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        directions_target_exp = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        return directions_target_exp

    def get_emotion_feature(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp) # torch.Size([72, 40])
        # alpha_D_exp = alpha_D_exp.mean(0).unsqueeze(0)
        # e = self.direction_exp.only_exp(alpha_D_exp) #torch.Size([1, 512])
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        directions_target_exp = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        return directions_target_exp


    def only_exp_2(self, img_source, exp_img_drive, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(exp_img_drive)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    def only_exp_3_1(self, img_source, e, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    # no mean
    def only_exp_4(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)


        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        e = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    
    def only_exp_from_pth(self, img_source, e, h_start=None):
        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)


        # alpha_target = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        # directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        # e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats, None)
        return img_recon

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio_lip(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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

        alpha_D_lip_non_nonlip = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D_lip_non_nonlip = self.direction_lipnonlip(alpha_D_lip_non_nonlip)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])

        latent_poseD = wa + e + directions_D_lip_non_nonlip
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio_only_lip(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        alpha_D_lip_non_nonlip = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D_lip_non_nonlip = self.direction_lipnonlip(alpha_D_lip_non_nonlip)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])

        latent_poseD = wa + directions_D_lip_non_nonlip
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_exp_audio_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    def img_smooth(self, wa, feats, directions_D, e, h_start=None):
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def get_feat(self, img_source):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        return wa, feats

    def get_audio_feature(self, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa_t_p, wa_t_exp, _, _ = self.enc(pose_img_drive, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
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
        # latent_poseD = wa + directions_D 
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return directions_D, e

    def get_exp_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        return alpha_D_exp

class Generator_using_ADAIN2(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_using_ADAIN2, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_with_ADAIN(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_exp_0(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)
        alpha_D_exp = alpha_D_exp.mean(0)
        # e = self.direction_exp.only_exp(alpha_D_exp)
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return alpha_D_exp

    def only_exp_3_0(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp) # torch.Size([72, 40])
        alpha_D_exp = alpha_D_exp.mean(0).unsqueeze(0)
        # e = self.direction_exp.only_exp(alpha_D_exp) #torch.Size([1, 512])
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        directions_target_exp = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        return directions_target_exp

    def get_emotion_feature(self, exp_img_drive, h_start=None):

        wa_t_exp, _, _, _ = self.enc(exp_img_drive,None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp) # torch.Size([72, 40])
        # alpha_D_exp = alpha_D_exp.mean(0).unsqueeze(0)
        # e = self.direction_exp.only_exp(alpha_D_exp) #torch.Size([1, 512])
        # # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        # latent_poseD = wa + e
        # img_recon = self.dec(latent_poseD, None, feats, e)
        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        directions_target_exp = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        return directions_target_exp


    def only_exp_2(self, img_source, exp_img_drive, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(exp_img_drive)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    def only_exp_3_1(self, img_source, e, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    # no mean
    def only_exp_4(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)


        alpha_target = torch.cat([alpha_D_exp[:,20:], alpha_D_exp[:,34:], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        e = self.direction_exp.get_exp_latent(directions_target_share)
        # print(e==directions_target_exp)

        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    
    def only_exp_from_pth(self, img_source, e, h_start=None):
        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)


        # alpha_target = torch.cat([torch.cat([alpha_D_exp,alpha_D_exp],-1), alpha_D_exp[:,:6], alpha_D_exp], dim=-1) # torch.Size([1, 66])
        # directions_target_share = self.direction_exp.get_shared_out(alpha_target, self.direction_lipnonlip.weight) # torch.Size([1, 66, 512])  
        # e = self.direction_exp.get_exp_latent(directions_target_share)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats, None)
        return img_recon

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio_emotion_inter(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive1, exp_img_drive2, h_start=None,weight = 1):

        wa, wa_t_exp1, feats, feats_t = self.enc(img_source, exp_img_drive1, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp2, _,_ = self.enc(pose_img_drive, exp_img_drive2) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp1 = self.fc(wa_t_exp1)
        alpha_D_exp1 = self.exp_fc(shared_fc_exp1)

        shared_fc_exp2 = self.fc(wa_t_exp2)
        alpha_D_exp2 = self.exp_fc(shared_fc_exp2)

        alpha_D_exp = weight * alpha_D_exp1 + (1-weight) * alpha_D_exp2
        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_evaluation_using_npy(self, wa, feats, alpha_D_lip,  alpha_D_pose, alpha_D_exp, h_start=None):

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon
    
    def test_evaluation(self, img_source, lip_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        # alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_exp)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        alpha_D_lip = self.lip_fc(shared_fc_p)

        # shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio_lip(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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

        alpha_D_lip_non_nonlip = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D_lip_non_nonlip = self.direction_lipnonlip(alpha_D_lip_non_nonlip)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])

        latent_poseD = wa + e + directions_D_lip_non_nonlip
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio_only_lip(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        alpha_D_lip_non_nonlip = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D_lip_non_nonlip = self.direction_lipnonlip(alpha_D_lip_non_nonlip)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])

        latent_poseD = wa + directions_D_lip_non_nonlip
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_exp_audio_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon


    def img_smooth(self, wa, feats, directions_D, e, h_start=None):
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def get_feat(self, img_source):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        return wa, feats

    def get_audio_feature(self, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa_t_p, wa_t_exp, _, _ = self.enc(pose_img_drive, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
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
        # latent_poseD = wa + directions_D 
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return directions_D, e

    def get_ex_feature(self, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa_t_p, wa_t_exp, _, _ = self.enc(pose_img_drive, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
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

        return e

    def get_exp_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        return alpha_D_exp

class Generator_using_warp_ADAIN(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_using_warp_ADAIN, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_with_warp_ADAIN(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_exp(self, img_source, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, _ = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + e
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def only_source(self, img_source, h_start=None):

        wa, _, feats, _ = self.enc(img_source, None, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])


        # shared_fc_exp = self.fc(wa_t_exp)
        # alpha_D_exp = self.exp_fc(shared_fc_exp)

        # e = self.direction_exp.only_exp(alpha_D_exp)
        # directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa
        img_recon = self.dec(latent_poseD, None, feats, None)
        return img_recon

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def img_smooth(self, wa, feats, directions_D, e, h_start=None):
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def get_feat(self, img_source):

        wa, _, feats, _ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        return wa, feats

    def get_audio_feature(self, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa_t_p, wa_t_exp, _, _ = self.enc(pose_img_drive, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
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
        # latent_poseD = wa + directions_D 
        # img_recon = self.dec(latent_poseD, None, feats, e)
        return directions_D, e

    def get_exp_feature(self, img_source):

        # wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(img_source, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_exp = self.exp_fc(shared_fc_p)

        return alpha_D_exp

def lip_motion_decorrelation_loss(A, B):
    batch_size, D = A.size()
    
    # 计算特征之间的相关性矩阵
    correlation_matrix = torch.matmul(A.t(), B)  # A的转置乘以B
    
    # 计算相关性的平方并求和
    squared_correlation = torch.square(correlation_matrix).sum()
    
    # 计算损失
    loss = squared_correlation / D
    
    return loss

class Discor_Bank(nn.Module):
    def __init__(self, style_dim=512,  K=512):
        super(Discor_Bank, self).__init__()

        self.K = K
        self.register_buffer("exp_queue", torch.randn(K, style_dim))
        self.exp_queue = nn.functional.normalize(self.exp_queue, dim=0)

        # self.register_buffer("pose_queue", torch.randn(style_dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("lip_queue", torch.randn(K, style_dim))
        self.lip_queue = nn.functional.normalize(self.lip_queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, lip_feature, exp_feature):
        batch_size = lip_feature.shape[0]
        # self._dequeue_and_enqueue(current_lip_feature, )
        current_lip_feature = torch.cat([self.lip_queue.clone().detach(), lip_feature], dim=0)
        current_exp_feature = torch.cat([self.exp_queue.clone().detach(), exp_feature], dim=0)
        loss = lip_motion_decorrelation_loss(exp_feature, lip_feature)
        self.lip_queue = current_lip_feature[batch_size:].clone().detach()
        self.exp_queue = current_exp_feature[batch_size:].clone().detach()
        return loss
        

class Generator_using_EAM2(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_using_EAM2, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis_with_EAM2(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
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
    

    def test_exp(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon

    def test_exp_audio(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

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
        img_recon = self.dec(latent_poseD, None, feats, e)
        return img_recon