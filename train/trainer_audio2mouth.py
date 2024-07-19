import torch

import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from collections import OrderedDict
from networks_Lip_NonLip.discriminator import Discriminator
from networks_Lip_NonLip.generator import Generator

from networks_audio2lip.bilinear import crop_bbox_batch
from networks_audio2lip.syncnet import SyncNet_color as SyncNet
from networks_audio2lip.audio_encoder import Audio2Lip
import numpy as np

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.dis_weight = args.dis_weight
        self.audio2lip = Audio2Lip().to(device)
        self.train_generator = args.train_generator

        requires_grad(self.audio2lip.audio_encoder, False)

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.channel_multiplier).to(
            device)
        if self.train_generator:
            requires_grad(self.gen, False)
            requires_grad(self.gen.dec, True)
        if self.dis_weight !=0:
            self.dis = Discriminator(args.size, args.channel_multiplier).to(device)

        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        if args.distributed:
            self.audio2lip = nn.parallel.DistributedDataParallel(
                self.audio2lip,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
                )
            if self.train_generator:
                self.gen = nn.parallel.DistributedDataParallel(
                    self.gen,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                    )
            # self.dis = nn.parallel.DistributedDataParallel(
            #     self.dis,
            #     device_ids=[args.local_rank],
            #     output_device=args.local_rank,
            #     broadcast_buffers=False,
            #     find_unused_parameters=True,
            #     )
        # distributed computing
        # self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        # self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        # d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        if args.distributed:
            self.audio2lip = self.audio2lip.module
            if self.train_generator:
                self.gen = self.gen.module
                # self.dis = self.dis.module


        net_parameters = filter(lambda p: p.requires_grad, self.audio2lip.parameters())
        train_parameters = list(net_parameters)
        if self.train_generator:
            net_parameters2 = filter(lambda p: p.requires_grad, self.gen.parameters())
            train_parameters += list(net_parameters2)
        self.g_optim = optim.Adam(
            train_parameters,
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )



        # self.d_optim = optim.Adam(
        #     self.dis.parameters(),
        #     lr=args.lr * d_reg_ratio,
        #     betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        # )

        # self.criterion_vgg = VGGLoss().to('cuda')

        self.criterion_vgg = VGGLoss().to('cuda')

        self.sync_weight = args.sync_weight
        
        if self.sync_weight != 0:
            self.syncnet = SyncNet()
            syncnet_state = torch.load('ckpt/checkpoint.pth', map_location='cpu')
            s = syncnet_state["state_dict"]
            self.syncnet.load_state_dict(s)
            for p in self.syncnet.parameters():
                p.requires_grad = False
            if torch.cuda.is_available():
                self.syncnet = self.syncnet.cuda()
                self.syncnet.eval()
            self.logloss = nn.BCELoss()

        self.start_iter = 0
    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, audio_features, lip_features, pose_features, identity_img, target_img, bbox = None): # torch.Size([64, 5, 80, 16])
        
        self.audio2lip.train()
        self.gen.train()
        self.gen.zero_grad()
        self.audio2lip.zero_grad()
        G_losses = {}
        # requires_grad(self.audio2lip, True)
        # requires_grad(self.gen.enc, False)
        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        # requires_grad(self.audio2lip.audio_encoder, False)
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = bi['img_a_identity'],bi['img_b_identity'],bi['img_a'],bi['img_b'],bi['imagea_b'],bi['imageb_a']
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = img_a_identity.cuda(), img_b_identity.cuda(), img_a.cuda(), img_b.cuda(), imagea_b.cuda(), imageb_a.cuda()
        batch_size, T = audio_features.shape[0], audio_features.shape[1]
        audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
        lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
        # .reshape([bbs*bs, 3, 256, 256]) 
        G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
        G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.1

        wa_identity, _, feats_identity, _ = self.gen.enc(identity_img) 
        lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
        pose_features = pose_features.reshape([batch_size*T, 6])
        alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
        directions = self.gen.direction_lipnonlip(alpha)
        rep = torch.LongTensor([T]*batch_size).cuda()
        wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
        layer_num = len(feats_identity)
        for i in range(layer_num):
            feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

        latent = wa_identity + directions

        recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])
        if self.dis_weight !=0:
            recon_pred = self.dis(recon)
        target_img = target_img.reshape([batch_size*T, 3, 256, 256]) 

        G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean()
        if self.dis_weight !=0:
            G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

        G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img)
        
        if self.sync_weight != 0:

            preds = bbox.reshape(batch_size*T, 4)
            preds = preds.to('cuda')/256.
            box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
            gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
            pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

            G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean()
            pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4) # torch.Size([20, 1, 80, 16])
            value = self.get_sync_loss(audio_features.reshape(batch_size, T, 80,16)[:,0:1], pre_bbox, 'cuda').mean() # torch.Size([4, 3, 5, 96, 96])
            G_losses['sync'] = self.sync_weight * value
            
        G_losses_values = [val.mean() for val in G_losses.values()]
        g_loss = sum(G_losses_values)

        g_loss.backward()
        self.g_optim.step()
        recon = recon.reshape(batch_size, T, 3,256,256)
        return G_losses, recon[:,-1], g_loss

    def get_sync_loss(self, mel, g, device):
        def cosine_loss(a, v, y): # torch.Size([4, 512]) y torch.Size([4, 1])
            d = F.cosine_similarity(a, v)
            with torch.cuda.amp.autocast(enabled=False):
                loss = self.logloss(d.unsqueeze(1), y)
            return loss
        g = g[:, :, :, g.size(3)//2:] # # torch.Size([4, 3, 5, 96, 96])
        g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=1) # torch.Size([4, 15, 48, 96])
        # B, 3 * T, H//2, W
        a, v = self.syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        return cosine_loss(a, v, y)


    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, audio_features, lip_features, pose_features, identity_img, target_img, bbox):
        with torch.no_grad():
            self.audio2lip.eval()
            G_losses = {}

            batch_size, T = audio_features.shape[0], audio_features.shape[1]
            audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
            lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
            # .reshape([bbs*bs, 3, 256, 256]) 
            G_losses['recon_l2_loss'] = F.mse_loss(lip_features_predict, lip_features)
            G_losses['recon_smooth'] = F.mse_loss(lip_features_predict[:,1:]-lip_features_predict[:,:-1], lip_features[:,1:]-lip_features[:,:-1])*0.1

            wa_identity, _, feats_identity, _ = self.gen.enc(identity_img) 
            lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
            pose_features = pose_features.reshape([batch_size*T, 6])
            alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            rep = torch.LongTensor([T]*batch_size).cuda()
            wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
            layer_num = len(feats_identity)
            for i in range(layer_num):
                feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

            latent = wa_identity + directions

            recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])
            if self.dis_weight !=0:
                recon_pred = self.dis(recon)
            target_img = target_img.reshape([batch_size*T, 3, 256, 256]) 

            G_losses['recon_vgg_loss'] = self.criterion_vgg(recon, target_img).mean()
            if self.dis_weight !=0:
                G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred)

            G_losses['recon_l1_loss'] = F.l1_loss(recon, target_img)
            
            if self.sync_weight != 0:

                preds = bbox.reshape(batch_size*T, 4)
                preds = preds.to('cuda')/256.
                box_to_feat = torch.from_numpy(np.array([i for i in range(batch_size*T)]))
                gt_bbox = crop_bbox_batch(target_img, preds, box_to_feat, 96)
                pre_bbox = crop_bbox_batch(recon, preds, box_to_feat, 96)

                G_losses['img_l1_sync'] = torch.abs(gt_bbox-pre_bbox).mean()
                pre_bbox = pre_bbox.reshape(batch_size, T, 3, 96, 96).permute(0, 2, 1, 3, 4) # torch.Size([20, 1, 80, 16])
                value = self.get_sync_loss(audio_features.reshape(batch_size, T, 80,16)[:,0:1], pre_bbox, 'cuda').mean() # torch.Size([4, 3, 5, 96, 96])
                G_losses['sync'] = self.sync_weight * value
                
            G_losses_values = [val.mean() for val in G_losses.values()]
            g_loss = sum(G_losses_values)


            recon = recon.reshape(batch_size, T, 3,256,256)
            return G_losses, recon[:,-1], g_loss

    def sample_no_loss(self, audio_features, lip_features, pose_features, identity_img, target_img):
        with torch.no_grad():
            self.audio2lip.eval()
            G_losses = {}

            batch_size, T = audio_features.shape[0], audio_features.shape[1]
            audio_features = audio_features.view(-1, 80, 16).unsqueeze(dim=1)
            lip_features_predict = self.audio2lip(audio_features, batch_size, T) # batch, T, 20
            wa_identity, _, feats_identity, _ = self.gen.enc(identity_img) 
            lip_features_predict = lip_features_predict.reshape([batch_size*T, 20])
            pose_features = pose_features.reshape([batch_size*T, 6])
            alpha = torch.cat([lip_features_predict, pose_features], dim=-1)
            directions = self.gen.direction_lipnonlip(alpha)
            rep = torch.LongTensor([T]*batch_size).cuda()
            wa_identity = torch.repeat_interleave(wa_identity, rep, dim=0)
            layer_num = len(feats_identity)
            for i in range(layer_num):
                feats_identity[i] = torch.repeat_interleave(feats_identity[i], rep, dim=0)

            latent = wa_identity + directions

            recon = self.gen.dec(latent, None, feats_identity)# torch.Size([20, 3, 256, 256])

            recon = recon.reshape(batch_size, T, 3,256,256)
            return recon[:,-1]

    def resume(self, resume_ckpt, audio2lip_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        # try:
        #     self.start_iter = ckpt["start_iter"] #int(os.path.splitext(ckpt_name)[0])
        # except:
        #     self.start_iter = 0
        try:
            start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            start_iter = 0
        # self.gen.load_state_dict(ckpt["gen"])

        checkpoint = ckpt['gen']
        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'enc.fc.' in key:
        #         if 'enc.fc.4' in key:
        #             continue
        #         name = key.split('enc.fc.')[1]
        #         new_state_dict[name] = value

        self.gen.load_state_dict(checkpoint)

        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'enc.net_app.' in key:
        #         name = key.split('enc.')[1]
        #         new_state_dict[name] = value
        # self.gen.enc.load_state_dict(new_state_dict)
        audio_encoder_ckpt = torch.load(audio2lip_ckpt)
        self.audio2lip.load_state_dict(audio_encoder_ckpt['audio2lip'])
        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'dec.' in key:
        #         if 'dec.direc' in key:
        #             continue
        #         name = key.split('dec.')[1]
        #         new_state_dict[name] = value
        # self.gen.dec.load_state_dict(new_state_dict)
        if self.dis_weight !=0:
            self.dis.load_state_dict(ckpt["dis"])
        try:
            self.g_optim.load_state_dict(ckpt["g_optim"])
        except:
            print('cannot load pretrained g_optim')
        # self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "audio2lip": self.audio2lip.state_dict(),
                # "dis": self.dis.state_dict(),
                # "g_optim": self.g_optim.state_dict(),
                # "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
