import torch
from networks_Lip_NonLip.discriminator import Discriminator
from networks_Lip_NonLip.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from collections import OrderedDict

def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


class Trainer(nn.Module):
    def __init__(self, args, device):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_lip, args.latent_dim_pose, args.channel_multiplier).to(
            device)
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)
        # requires_grad(self.gen.enc, False)
        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        if args.distributed:
            self.gen = nn.parallel.DistributedDataParallel(
                self.gen,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
                )
            self.dis = nn.parallel.DistributedDataParallel(
                self.dis,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
                )
        # distributed computing
        # self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        # self.dis = DDP(self.dis, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        if args.distributed:
            self.gen = self.gen.module
            self.dis = self.dis.module


        net_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())

        self.g_optim = optim.Adam(
            net_parameters,
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to('cuda')

        self.criterion_vgg = VGGLoss().to('cuda')

        self.start_iter = 0
    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a):
        self.gen.train()
        self.gen.zero_grad()
        G_losses = {}
        requires_grad(self.gen, True)
        # requires_grad(self.gen.enc, False)
        # requires_grad(self.gen.dec, False)
        # requires_grad(self.gen.fc, False)
        requires_grad(self.dis, False)
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = bi['img_a_identity'],bi['img_b_identity'],bi['img_a'],bi['img_b'],bi['imagea_b'],bi['imageb_a']
        # img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = img_a_identity.cuda(), img_b_identity.cuda(), img_a.cuda(), img_b.cuda(), imagea_b.cuda(), imageb_a.cuda()
        
        wa_a_identity, wa_b_identity, feats_a_identity, feats_b_identity = self.gen.enc(img_a_identity, img_b_identity) 
        wa_a, wa_b, _, _ = self.gen.enc(img_a, img_b) 
        wa_a_b, wa_b_a, _, _ = self.gen.enc(imagea_b, imageb_a) 

        # recon: a,b
        shared_fc_a = self.gen.fc(wa_a)
        lip_a = self.gen.lip_fc(shared_fc_a) # torch.Size([12, 20])
        pose_a = self.gen.pose_fc(shared_fc_a) # torch.Size([12, 6])
        alpha_a = torch.cat([lip_a, pose_a], dim=-1)
        directions_a_share = self.gen.direction_lipnonlip.get_shared_out(alpha_a)
        directions_a_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_a_share)
        directions_a_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_a_share)
        directions_a = directions_a_lip+directions_a_pose # torch.Size([12, 512])
        latent_a = wa_a_identity + directions_a
        recon_a = self.gen.dec(latent_a, None, feats_a_identity)
        # recon_a_pred = self.dis(recon_a)

        shared_fc_b = self.gen.fc(wa_b)
        lip_b = self.gen.lip_fc(shared_fc_b)
        pose_b = self.gen.pose_fc(shared_fc_b)
        alpha_b = torch.cat([lip_b, pose_b], dim=-1)

        directions_b_share = self.gen.direction_lipnonlip.get_shared_out(alpha_b)
        directions_b_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_b_share)
        directions_b_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_b_share)
        directions_b = directions_b_lip+directions_b_pose

        # directions_b = self.gen.direction_lipnonlip(alpha_b)
        latent_b = wa_b_identity + directions_b
        recon_b = self.gen.dec(latent_b, None, feats_b_identity)
        # recon_b_pred = self.dis(recon_b)

        G_losses['recon_vgg_loss'] = self.criterion_vgg(recon_a, img_a).mean() + self.criterion_vgg(recon_b, img_b).mean()
        G_losses['recon_l1_loss'] = F.l1_loss(recon_a, img_a) + F.l1_loss(recon_b, img_b)
        # G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_a_pred) + self.g_nonsaturating_loss(recon_b_pred)

        # cross_recon: a,b
        shared_fc_a_b = self.gen.fc(wa_a_b)
        lip_a_b = self.gen.lip_fc(shared_fc_a_b) # b的lip
        pose_a_b = self.gen.pose_fc(shared_fc_a_b) # a的pose

        shared_fc_b_a = self.gen.fc(wa_b_a)
        lip_b_a = self.gen.lip_fc(shared_fc_b_a) # a的lip
        pose_b_a = self.gen.pose_fc(shared_fc_b_a) # b的pose

        # start cross
        alpha_a_cross = torch.cat([lip_b_a, pose_a_b], dim=-1) # cross recon a
        # directions_a_cross = self.gen.direction_lipnonlip(alpha_a_cross)

        directions_a_cross_share = self.gen.direction_lipnonlip.get_shared_out(alpha_a_cross)
        directions_a_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_a_cross_share)
        directions_a_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_a_cross_share)
        directions_a_cross = directions_a_cross_lip+directions_a_cross_pose

        latent_a_corss = wa_a_identity + directions_a_cross
        recon_a_cross = self.gen.dec(latent_a_corss, None, feats_a_identity)
        # recon_a_pred_cross = self.dis(recon_a_cross)

        alpha_b_cross = torch.cat([lip_a_b, pose_b_a], dim=-1) # cross recon b
        # directions_b_cross = self.gen.direction_lipnonlip(alpha_b_cross)

        directions_b_cross_share = self.gen.direction_lipnonlip.get_shared_out(alpha_b_cross)
        directions_b_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_b_cross_share)
        directions_b_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_b_cross_share)
        directions_b_cross = directions_b_cross_lip+directions_b_cross_pose


        latent_b_corss = wa_b_identity + directions_b_cross
        recon_b_cross = self.gen.dec(latent_b_corss, None, feats_b_identity)
        # recon_b_pred_cross = self.dis(recon_b_cross)

        recon_pred_total = self.dis(torch.cat([recon_a, recon_b, recon_a_cross, recon_b_cross]))

        G_losses['cross_recon_vgg_loss'] = self.criterion_vgg(recon_a_cross, img_a).mean() + self.criterion_vgg(recon_b_cross, img_b).mean()
        G_losses['cross_recon_l1_loss'] = F.l1_loss(recon_a_cross, img_a) + F.l1_loss(recon_b_cross, img_b)
        G_losses['cross_recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred_total)

        # latent loss
        # lip space
        G_losses['lip_space'] = torch.exp(-F.cosine_similarity(directions_b_cross_lip, directions_b_lip))+torch.exp(-F.cosine_similarity(directions_a_cross_lip, directions_a_lip))
        G_losses['pose_space'] = torch.exp(-F.cosine_similarity(directions_b_cross_pose, directions_b_pose))+torch.exp(-F.cosine_similarity(directions_a_cross_pose, directions_a_pose))
        G_losses['lip_space'] = G_losses['lip_space'].sum()
        G_losses['pose_space'] = G_losses['pose_space'].sum()
        # img_target_recon = self.gen(img_a_identity, img_a)
        # img_recon_pred = self.dis(img_target_recon)
        
        # vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
        # l1_loss = F.l1_loss(img_target_recon, img_target)
        # gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)
        G_losses_values = [val.mean() for val in G_losses.values()]
        g_loss = sum(G_losses_values)

        g_loss.backward()
        self.g_optim.step()

        return G_losses, recon_a_cross, recon_b_cross, g_loss

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

    def sample(self, img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a):
        with torch.no_grad():
            self.gen.eval()


            G_losses = {}


            wa_a_identity, wa_b_identity, feats_a_identity, feats_b_identity = self.gen.enc(img_a_identity, img_b_identity) 
            wa_a, wa_b, _, _ = self.gen.enc(img_a, img_b) 
            wa_a_b, wa_b_a, _, _ = self.gen.enc(imagea_b, imageb_a) 

            # recon: a,b
            shared_fc_a = self.gen.fc(wa_a)
            lip_a = self.gen.lip_fc(shared_fc_a) # torch.Size([12, 20])
            pose_a = self.gen.pose_fc(shared_fc_a) # torch.Size([12, 6])
            alpha_a = torch.cat([lip_a, pose_a], dim=-1)
            directions_a_share = self.gen.direction_lipnonlip.get_shared_out(alpha_a)
            directions_a_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_a_share)
            directions_a_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_a_share)
            directions_a = directions_a_lip+directions_a_pose # torch.Size([12, 512])
            latent_a = wa_a_identity + directions_a
            recon_a = self.gen.dec(latent_a, None, feats_a_identity)
            # recon_a_pred = self.dis(recon_a)

            shared_fc_b = self.gen.fc(wa_b)
            lip_b = self.gen.lip_fc(shared_fc_b)
            pose_b = self.gen.pose_fc(shared_fc_b)
            alpha_b = torch.cat([lip_b, pose_b], dim=-1)

            directions_b_share = self.gen.direction_lipnonlip.get_shared_out(alpha_b)
            directions_b_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_b_share)
            directions_b_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_b_share)
            directions_b = directions_b_lip+directions_b_pose

            # directions_b = self.gen.direction_lipnonlip(alpha_b)
            latent_b = wa_b_identity + directions_b
            recon_b = self.gen.dec(latent_b, None, feats_b_identity)
            # recon_b_pred = self.dis(recon_b)

            G_losses['recon_vgg_loss'] = self.criterion_vgg(recon_a, img_a).mean() + self.criterion_vgg(recon_b, img_b).mean()
            G_losses['recon_l1_loss'] = F.l1_loss(recon_a, img_a) + F.l1_loss(recon_b, img_b)
            # G_losses['recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_a_pred) + self.g_nonsaturating_loss(recon_b_pred)

            # cross_recon: a,b
            shared_fc_a_b = self.gen.fc(wa_a_b)
            lip_a_b = self.gen.lip_fc(shared_fc_a_b) # b的lip
            pose_a_b = self.gen.pose_fc(shared_fc_a_b) # a的pose

            shared_fc_b_a = self.gen.fc(wa_b_a)
            lip_b_a = self.gen.lip_fc(shared_fc_b_a) # a的lip
            pose_b_a = self.gen.pose_fc(shared_fc_b_a) # b的pose

            # start cross
            alpha_a_cross = torch.cat([lip_b_a, pose_a_b], dim=-1) # cross recon a
            # directions_a_cross = self.gen.direction_lipnonlip(alpha_a_cross)

            directions_a_cross_share = self.gen.direction_lipnonlip.get_shared_out(alpha_a_cross)
            directions_a_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_a_cross_share)
            directions_a_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_a_cross_share)
            directions_a_cross = directions_a_cross_lip+directions_a_cross_pose

            latent_a_corss = wa_a_identity + directions_a_cross
            recon_a_cross = self.gen.dec(latent_a_corss, None, feats_a_identity)
            # recon_a_pred_cross = self.dis(recon_a_cross)

            alpha_b_cross = torch.cat([lip_a_b, pose_b_a], dim=-1) # cross recon a
            # directions_b_cross = self.gen.direction_lipnonlip(alpha_b_cross)

            directions_b_cross_share = self.gen.direction_lipnonlip.get_shared_out(alpha_b_cross)
            directions_b_cross_lip = self.gen.direction_lipnonlip.get_lip_latent(directions_b_cross_share)
            directions_b_cross_pose = self.gen.direction_lipnonlip.get_pose_latent(directions_b_cross_share)
            directions_b_cross = directions_b_cross_lip+directions_b_cross_pose


            latent_b_corss = wa_b_identity + directions_b_cross
            recon_b_cross = self.gen.dec(latent_b_corss, None, feats_b_identity)
            # recon_b_pred_cross = self.dis(recon_b_cross)

            recon_pred_total = self.dis(torch.cat([recon_a, recon_b, recon_a_cross, recon_b_cross]))

            G_losses['cross_recon_vgg_loss'] = self.criterion_vgg(recon_a_cross, img_a).mean() + self.criterion_vgg(recon_b_cross, img_b).mean()
            G_losses['cross_recon_l1_loss'] = F.l1_loss(recon_a_cross, img_a) + F.l1_loss(recon_b_cross, img_b)
            G_losses['cross_recon_gan_g_loss'] = self.g_nonsaturating_loss(recon_pred_total)

            # latent loss
            # lip space
            G_losses['lip_space'] = torch.exp(-F.cosine_similarity(directions_b_cross_lip, directions_b_lip))+torch.exp(-F.cosine_similarity(directions_a_cross_lip, directions_a_lip))
            G_losses['pose_space'] = torch.exp(-F.cosine_similarity(directions_b_cross_pose, directions_b_pose))+torch.exp(-F.cosine_similarity(directions_a_cross_pose, directions_a_pose))
            G_losses['lip_space'] = G_losses['lip_space'].sum()
            G_losses['pose_space'] = G_losses['pose_space'].sum()
            # img_target_recon = self.gen(img_a_identity, img_a)
            # img_recon_pred = self.dis(img_target_recon)
            
            # vgg_loss = self.criterion_vgg(img_target_recon, img_target).mean()
            # l1_loss = F.l1_loss(img_target_recon, img_target)
            # gan_g_loss = self.g_nonsaturating_loss(img_recon_pred)
            G_losses_values = [val.mean() for val in G_losses.values()]
            g_loss = sum(G_losses_values)


            return G_losses, recon_a_cross, recon_b_cross, g_loss

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=torch.device('cpu'))
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

        # new_state_dict = OrderedDict()
        # for key, value in checkpoint.items():
        #     if 'dec.' in key:
        #         if 'dec.direc' in key:
        #             continue
        #         name = key.split('dec.')[1]
        #         new_state_dict[name] = value
        # self.gen.dec.load_state_dict(new_state_dict)

        self.dis.load_state_dict(ckpt["dis"])
        try:
            self.g_optim.load_state_dict(ckpt["g_optim"])
        except:
            print('cannot load pretrained g_optim')
        self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.state_dict(),
                "dis": self.dis.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
