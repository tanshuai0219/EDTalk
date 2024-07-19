import argparse

import datetime, os
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1'
import torch
from torch.utils import data
from torch.utils.data import DataLoader
# from dataset import Vox256, Taichi, TED
from datasets.dataset_HDTF_lip_nonlip import HDTF_LipNonLipDataset
import torchvision
import torchvision.transforms as transforms
from train.trainer_Mouth_Pose_decouple import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
# from config import Config
import torch.nn.functional as F
from torchvision import utils
import shutil
from util.distributed_stylegan import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore")

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def display_img(idx, img, name, writer):
    img = img.clamp(-1, 1)
    img = ((img - img.min()) / (img.max() - img.min())).data

    writer.add_images(tag='%s' % (name), global_step=idx, img_tensor=img)


def write_loss(i, vgg_loss, l1_loss, g_loss, d_loss, writer):
    writer.add_scalar('vgg_loss', vgg_loss.item(), i)
    writer.add_scalar('l1_loss', l1_loss.item(), i)
    writer.add_scalar('gen_loss', g_loss.item(), i)
    writer.add_scalar('dis_loss', d_loss.item(), i)
    writer.flush()


def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.addr
    os.environ['MASTER_PORT'] = args.port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(args):
    # init distributed computing
    # ddp_setup(args, rank, world_size)
    # torch.cuda.set_device(rank)
    device = torch.device("cuda")
    date = str(datetime.datetime.now())
    date = date[:date.rfind(":")].replace("-", "")\
                                .replace(":", "")\
                                .replace(" ", "_")
    # make logging folder
    log_path = os.path.join('logs', args.exp_path, args.exp_name, date, 'log')
    checkpoint_path = os.path.join('logs', args.exp_path, args.exp_name, date, 'checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    best_loss = 10000
    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    # if args.dataset == 'ted':
    #     dataset = TED('train', transform, True)
    #     dataset_test = TED('test', transform)
    # elif args.dataset == 'vox':
    dataset = HDTF_LipNonLipDataset(args, is_inference= False, transform = transform)
    dataset_test = HDTF_LipNonLipDataset(args, is_inference= True, transform = transform)
    # elif args.dataset == 'taichi':
    #     dataset = Taichi('train', transform, True)
    #     dataset_test = Taichi('test', transform)
    # else:
    #     raise NotImplementedError

    if args.distributed == False:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler, pin_memory=True, drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=(test_sampler is None), num_workers=4, sampler=test_sampler, pin_memory=True, drop_last=True)

    trainSteps = len(dataloader)
    testSteps = len(test_dataloader)
    # loader = sample_data(loader)
    # loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    current_iter = args.start_iter
    test_iter = 0
    print('==> training')
    # pbar = range(args.iter)
    last_name = None
    test_epoch= 0
    for epoch in range(args.epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        for bii, bi in enumerate(dataloader):
            current_iter += 1
            img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = bi['img_a_identity'],bi['img_b_identity'],bi['img_a'],bi['img_b'],bi['imagea_b'],bi['imageb_a']
            img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = img_a_identity.cuda(), img_b_identity.cuda(), img_a.cuda(), img_b.cuda(), imagea_b.cuda(), imageb_a.cuda()
            
            # update generator
            G_losses, recon_a_cross, recon_b_cross, l_g_total = trainer.gen_update(img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a)

            # update discriminator
            gan_d_loss = trainer.dis_update(torch.cat([img_a, img_b], dim=0), torch.cat([recon_a_cross, recon_b_cross], dim=0))

            log_info = 'train_Epoch [{}/{}], Step [{}/{}] '.format(epoch, args.epoch, current_iter, args.iter,
                        l_g_total.detach().item())
            
            for key, value in G_losses.items():
                log_info+='{}: {:.4f} '.format(key, value.detach().item())
            print(log_info)

            if current_iter % args.log_iter == 0:
                writer.add_scalar('train_loss/l_g_total_loss', l_g_total.detach().item(), current_iter)
                for key, value in G_losses.items():
                    writer.add_scalar('train_loss/{}'.format(key), value.detach().item(), current_iter)
            # display
            # if current_iter % args.display_freq == 0:
            #     print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f]"
            #         % (current_iter, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item()))
            if current_iter % args.image_save_iter == 0:

                sample = F.interpolate(torch.cat((img_a_identity.detach(),imagea_b.detach(), imageb_a.detach(), recon_a_cross.detach()), dim=0), 256)
                # sample = torch.flip(sample, [1])
                utils.save_image(
                    sample,
                    os.path.join(checkpoint_path, "epoch_%05d_step_%05d_train.jpg"%(epoch, current_iter)),
                    nrow=int(args.batch_size),
                    normalize=True,
                    range=(-1, 1),
                )

            if current_iter % args.eval_iter == 0:
                curloss = 0
                # lip_mlp = lip_mlp.eval()
                if args.distributed:
                    test_epoch +=1
                    test_dataloader.sampler.set_epoch(test_epoch)
                for bii, bi in enumerate(test_dataloader):
                    test_iter+=1
                    with torch.no_grad():
                        img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = bi['img_a_identity'],bi['img_b_identity'],bi['img_a'],bi['img_b'],bi['imagea_b'],bi['imageb_a']
                        img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a = img_a_identity.cuda(), img_b_identity.cuda(), img_a.cuda(), img_b.cuda(), imagea_b.cuda(), imageb_a.cuda()
                        

                        G_losses, recon_a_cross, recon_b_cross, l_g_total = trainer.sample(img_a_identity, img_b_identity, img_a, img_b, imagea_b, imageb_a)
                        curloss+= l_g_total.detach().item()

                        log_info = 'eval_Epoch [{}/{}], Step [{}/{}] '.format(epoch, args.epoch, bii, testSteps,
                                    l_g_total.detach().item())
                        
                        for key, value in G_losses.items():
                            log_info+='{}: {:.4f} '.format(key, value.detach().item())
                        print(log_info)

                        if test_iter % args.log_iter == 0:
                            writer.add_scalar('eval_loss/l_g_total_loss', l_g_total.detach().item(), test_iter)
                            for key, value in G_losses.items():
                                writer.add_scalar('eval_loss/{}'.format(key), value.detach().item(), test_iter)
                        if test_iter % args.image_save_iter == 0:
                            sample = F.interpolate(torch.cat((img_a_identity.detach(),imagea_b.detach(), imageb_a.detach(), recon_a_cross.detach()), dim=0), 256)
                            # sample = torch.flip(sample, [1])
                            utils.save_image(
                                sample,
                                os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, test_iter)),
                                nrow=int(args.batch_size),
                                normalize=True,
                                range=(-1, 1),
                            )
                            last_name = os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, test_iter))
                        # break
                curloss = curloss/len(test_dataloader)


                if curloss<best_loss:
                    best_loss = curloss
                    trainer.save(current_iter, checkpoint_path)
                    if last_name!=None:
                        shutil.copy(last_name, os.path.join(checkpoint_path, 'best_epoch_%06d_step_%06d.jpg'%(epoch, current_iter)))
            
            if current_iter%args.save_freq==0:
                trainer.save(current_iter, checkpoint_path)
                if last_name!=None:
                    shutil.copy(last_name, os.path.join(checkpoint_path, 'step_%06d.jpg'%(current_iter)))
            # # save model
            # if i % args.save_freq == 0 and rank == 0:
            #     trainer.save(i, checkpoint_path)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_lip", type=int, default=20)
    parser.add_argument("--latent_dim_pose", type=int, default=6)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./HDTF_lipnonlip/')
    parser.add_argument("--exp_name", type=str, default='v3')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--semantic_radius", type=int, default=13)
    parser.add_argument("--log_iter", type=int, default=1)
    parser.add_argument("--image_save_iter", type=int, default=100, help="local rank for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--distributed", type=bool, default=False, help="local rank for distributed training")
    parser.add_argument("--eval_iter", type=int, default=1000, help="local rank for distributed training")
    
    opts = parser.parse_args()

    # opts = Config(opts.config, opts, is_train=True)
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opts.distributed = n_gpu > 1
    # n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2

    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    main(opts)