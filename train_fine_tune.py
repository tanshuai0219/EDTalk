import argparse
import os
import torch
from torch.utils import data
from fine_tune.dataset import Finetune256
import torchvision
import torchvision.transforms as transforms
from fine_tune.trainer_fine_tune import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchvision import utils

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


def main(rank, world_size, args):
    # init distributed computing
    ddp_setup(args, rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda")

    # make logging folder
    log_path = os.path.join(args.exp_path, args.exp_name + '/log')
    checkpoint_path = os.path.join(args.exp_path, args.exp_name + '/checkpoint')

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    writer = SummaryWriter(log_path)

    print('==> preparing dataset')
    transform = torchvision.transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    dataset = Finetune256('/data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop_frame/Obama', True, transform = transform)
    dataset_test = Finetune256('/data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop_frame/Obama', False, transform = transform)
    loader = data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=args.batch_size // world_size,
        sampler=data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
        drop_last=False,
    )

    loader_test = data.DataLoader(
        dataset_test,
        num_workers=8,
        batch_size=4,
        sampler=data.distributed.DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False),
        pin_memory=True,
        drop_last=False,
    )

    loader = sample_data(loader)
    loader_test = sample_data(loader_test)

    print('==> initializing trainer')
    # Trainer
    trainer = Trainer(args, device, rank)

    # resume
    if args.resume_ckpt is not None:
        args.start_iter = trainer.resume(args.resume_ckpt)
        print('==> resume from iteration %d' % (args.start_iter))

    print('==> training')
    pbar = range(args.iter)
    for idx in pbar:
        i = idx + args.start_iter

        # laoding data
        img_source, img_target = next(loader)
        img_source = img_source.to(rank, non_blocking=True)
        img_target = img_target.to(rank, non_blocking=True)

        # update generator
        vgg_loss, l1_loss, gan_g_loss, img_recon = trainer.gen_update(img_source, img_target)

        # update discriminator
        gan_d_loss = trainer.dis_update(img_target, img_recon)

        if rank == 0:
            # write to log
            write_loss(idx, vgg_loss, l1_loss, gan_g_loss, gan_d_loss, writer)

        # display
        if i % args.display_freq == 0 and rank == 0:
            print("[Iter %d/%d] [vgg loss: %f] [l1 loss: %f] [g loss: %f] [d loss: %f]"
                  % (i, args.iter, vgg_loss.item(), l1_loss.item(), gan_g_loss.item(), gan_d_loss.item()))

            if rank == 0:
                img_test_source, img_test_target = next(loader_test)
                img_test_source = img_test_source.to(rank, non_blocking=True)
                img_test_target = img_test_target.to(rank, non_blocking=True)

                img_recon = trainer.sample(img_test_source, img_test_target)
                
                sample = F.interpolate(torch.cat((img_test_source.detach(),img_test_target.detach(), img_recon.detach()), dim=0), 256)
                # sample = torch.flip(sample, [1])
                utils.save_image(
                    sample,
                    os.path.join(checkpoint_path, "step_%05d.jpg"%(i)),
                    nrow=4,
                    normalize=True,
                    range=(-1, 1),
                )

                # display_img(i, img_test_source, 'source', writer)
                # display_img(i, img_test_target, 'target', writer)
                # display_img(i, img_recon, 'recon', writer)
                # display_img(i, img_source_ref, 'source_ref', writer)
                # writer.flush()

        # save model
        if i % args.save_freq == 0 and rank == 0:
            trainer.save(i, checkpoint_path)

    return


if __name__ == "__main__":
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--only_fine_tune_dec", action='store_true', help='Only fine tune dec in Generator')
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--resume_ckpt", type=str, default='ckpts/EDTalk_lip_pose.pt')
    parser.add_argument("--datapath", type=str, default='/data/ts/datasets/person_specific_dataset/AD-NeRF/video_crop_frame/Obama')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=20)
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--exp_path", type=str, default='/data/ts/checkpoints/EDTalk/fine_tune/')
    parser.add_argument("--exp_name", type=str, default='Obama')
    parser.add_argument("--addr", type=str, default='localhost')
    parser.add_argument("--port", type=str, default='12345')
    opts = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2

    world_size = n_gpus
    print('==> training on %d gpus' % n_gpus)
    mp.spawn(main, args=(world_size, opts,), nprocs=world_size, join=True)