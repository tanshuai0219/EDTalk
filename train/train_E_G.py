import argparse
import os
import torch
from torch.utils import data
from torch.utils.data import DataLoader
# from dataset import Vox256, Taichi, TED
from datasets.dataset_MEAD_HDTF import VoxDataset
import torchvision
import torchvision.transforms as transforms
from train.trainer_E_G import Trainer
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

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

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
    dataset = VoxDataset(args, is_inference= False, transform = transform)
    dataset_test = VoxDataset(args, is_inference= True, transform = transform)
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
    print('==> training')
    # pbar = range(args.iter)
    last_name = None
    test_epoch= 0
    for epoch in range(args.epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        for bii, each_data in enumerate(dataloader):
            current_iter += 1
            img_source, img_target = each_data['source_image'], each_data['target_image']
            img_source = img_source.to(device)
            img_target = img_target.to(device)

            # update generator
            vgg_loss, l1_loss, gan_g_loss, img_recon = trainer.gen_update(img_source, img_target)

            # update discriminator
            gan_d_loss = trainer.dis_update(img_target, img_recon)

            if current_iter % args.log_iter == 0:
                # write to log
                write_loss(current_iter, vgg_loss, l1_loss, gan_g_loss, gan_d_loss, writer)

            # display
            if current_iter % args.display_freq == 0:
                print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f]"
                    % (current_iter, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item()))
            if current_iter % args.image_save_iter == 0:

                sample = F.interpolate(torch.cat((img_source.detach(),img_target.detach(), img_recon.detach()), dim=0), 256)
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
                    with torch.no_grad():
                        img_test_source, img_test_target = bi['source_image'], bi['target_image']

                        img_test_source = img_test_source.to(device)
                        img_test_target = img_test_target.to(device)

                        img_recon, img_source_ref = trainer.sample(img_test_source, img_test_target)
                        # curloss+= g_loss.detach().item()



                        sample = F.interpolate(torch.cat((img_test_source.detach(),img_test_target.detach(), img_recon.detach(),img_source_ref.detach()), dim=0), 256)
                        # sample = torch.flip(sample, [1])
                        utils.save_image(
                            sample,
                            os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, current_iter)),
                            nrow=int(args.batch_size),
                            normalize=True,
                            range=(-1, 1),
                        )
                        last_name = os.path.join(checkpoint_path, "epoch_%05d_step_%05d_test.jpg"%(epoch, current_iter))
                        break
                # curloss = curloss/len(test_dataloader)


                if curloss<best_loss:
                    best_loss = curloss
                    trainer.save(current_iter, checkpoint_path)
                    if last_name!=None:
                        shutil.copy(last_name, os.path.join(checkpoint_path, 'step_%06d.jpg'%(current_iter)))
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
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default='vox.pt')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=3000)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, default='vox')
    parser.add_argument("--exp_path", type=str, default='./MEAD_HDTF/')
    parser.add_argument("--exp_name", type=str, default='v1')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    parser.add_argument("--path", type=str, default='EDTalk_lmdb')
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--log_iter", type=int, default=10)
    parser.add_argument("--image_save_iter", type=int, default=1000, help="local rank for distributed training")
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