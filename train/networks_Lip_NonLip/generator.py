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
        if Q.dtype != torch.float32:
            Q = torch.tensor(Q, dtype=torch.float32)
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            if input_diag.dtype != torch.float32:
                input_diag = torch.tensor(input_diag, dtype=torch.float32)
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


class Pose_Direction(nn.Module):
    def __init__(self, pose_dim):
        super(Pose_Direction, self).__init__()
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, pose_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]
        if Q.dtype != torch.float32:
            Q = torch.tensor(Q, dtype=torch.float32)
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            if input_diag.dtype != torch.float32:
                input_diag = torch.tensor(input_diag, dtype=torch.float32)
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
    

class Lip_Direction(nn.Module):
    def __init__(self, lip_dim):
        super(Lip_Direction, self).__init__()
        self.lip_dim = lip_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim))

    def forward(self, input, pose_weight):
        # input: (bs*t) x 512
        weight = torch.cat([self.weight, pose_weight], -1)
        weight = weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]
        if Q.dtype != torch.float32:
            Q = torch.tensor(Q, dtype=torch.float32)
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

    def test_evaluation_using_npy_lip_pose(self, wa, feats, alpha_D_lip,  alpha_D_pose, h_start=None):

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D)
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon



    def test_evaluation_using_npy_lip(self, wa, feats, alpha_D_lip, h_start=None):

        alpha_D = torch.cat([alpha_D_lip, alpha_D_lip[:,:6]], dim=-1)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D)
        lip = self.direction_lipnonlip.get_lip_latent(directions_D)
        latent_poseD = wa + lip
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
        latent_poseD = wa + directions_D  # torch.Size([1, 512])
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_lip_nonlip_using_extract(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, _, feats, _ = self.enc(img_source, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        # alpha_D_lip = self.lip_fc(shared_fc)

        # shared_fc_p = self.fc(wa_t_p)
        # alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([lip_img_drive, pose_img_drive], dim=-1)
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
    
    def test_only_lip(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        lip_direction = self.direction_lipnonlip.get_lip_latent(directions_D)
        latent_poseD = wa + lip_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_manipulate_lip(self, img_source,i,j, h_start=None):

        wa, _, feats, _ = self.enc(img_source) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa)
        alpha_D_lip = self.lip_fc(shared_fc)

        # alpha_D_lip

        shared_fc_p = self.fc(wa)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        alpha_D[:,j] = alpha_D[:,j]*(1+0.1*i)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        lip_direction = self.direction_lipnonlip.get_lip_latent(directions_D)
        latent_poseD = wa + lip_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        # img_0 = self.dec(wa, None, feats)
        return img_recon#, img_0

    def test_manipulate_pose(self, img_source,i,j, h_start=None):

        wa, _, feats, _ = self.enc(img_source) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa)
        alpha_D_lip = self.lip_fc(shared_fc)

        # alpha_D_lip

        shared_fc_p = self.fc(wa)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        alpha_D[:,j] = alpha_D[:,j]*(1+0.1*i)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        lip_direction = self.direction_lipnonlip.get_pose_latent(directions_D)
        latent_poseD = wa + lip_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        # img_0 = self.dec(wa, None, feats)
        return img_recon#, img_0

    def test_only_pose(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        pose_direction = self.direction_lipnonlip.get_pose_latent(directions_D)
        latent_poseD = wa + pose_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_from_audio_pose_image(self, img_source, alpha_D_lip, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_pose = self.pose_fc(shared_fc)


        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_from_only_audio2lip(self, img_source, alpha_D_lip, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_pose = self.pose_fc(shared_fc)


        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
    
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        lip_direction = self.direction_lipnonlip.get_lip_latent(directions_D)
        latent_poseD = wa + lip_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon
class Generator_lip_nonlip(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator_lip_nonlip, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        # self.direction_lip = Lip_Direction(lip_dim)
        self.direction_pose = Pose_Direction(pose_dim)
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

    def test_lip_nonlip_using_extract(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, _, feats, _ = self.enc(img_source, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        # alpha_D_lip = self.lip_fc(shared_fc)

        # shared_fc_p = self.fc(wa_t_p)
        # alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([lip_img_drive, pose_img_drive], dim=-1)
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
    def test_only_lip(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        lip_direction = self.direction_lipnonlip.get_lip_latent(directions_D)
        latent_poseD = wa + lip_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_only_pose(self, img_source, lip_img_drive, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, feats_t_p,_ = self.enc(pose_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip.get_shared_out(alpha_D) # torch.Size([1, 512])
        pose_direction = self.direction_lipnonlip.get_pose_latent(directions_D)
        latent_poseD = wa + pose_direction 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon

    def test_from_audio_pose_image(self, img_source, alpha_D_lip, pose_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_pose = self.pose_fc(shared_fc)


        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose], dim=-1)
        directions_D = self.direction_lipnonlip(alpha_D) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, None, feats)
        return img_recon