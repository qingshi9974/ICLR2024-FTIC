from utils.datasets import TestKodakDataset,Datasets
import argparse
import math
import random
from utils.Meter import AverageMeterTEST,AverageMeterTRAIN
import sys
import time
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models import FrequencyAwareTransFormer
from pytorch_msssim import ms_ssim

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["ssim"] =1- ms_ssim(output["x_hat"], target,data_range=1.)
        out["loss"] = self.lmbda*255*255*out["mse_loss"]+out["bpp_loss"]
        return out




class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

  

    
def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeterTEST()
    bpp_loss = AverageMeterTEST()
    mse_loss = AverageMeterTEST()
    ssim_loss =AverageMeterTEST()
    aux_loss = AverageMeterTEST()
    bpp_z_loss = AverageMeterTEST()
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            N,_,H,W = d.size()
            num_pixels = N*H*W
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            bpp_z = (torch.log(out_net["likelihoods"]['z']).sum()/ (-math.log(2) * num_pixels))
    
            loss.update(out_criterion["loss"])
            mse_losss=out_criterion["mse_loss"]
            ssim_losss = -10 * math.log10(out_criterion['ssim'])

            psnr = 10 * (torch.log(1/ mse_losss) / np.log(10))
            mse_loss.update(psnr)
            bpp_z_loss.update(bpp_z)
            ssim_loss.update(ssim_losss)
      

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg :.3f} |"
        f"\tSSIM loss: {ssim_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tBpp z loss: {bpp_z_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256,256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--test", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpt/model.pth.tar", help="Where to Save model"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

elapsed,data_times, losses, psnrs, bpps, bpp_ys, bpp_zs, mse_losses,aux_losses = [AverageMeterTRAIN(2000) for _ in range(9)]


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_dataset = Datasets(data_dir='./training_set_path', transforms=train_transforms)
    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=False,
                              num_workers=8)
    test_dataset = TestKodakDataset(data_dir='./kodak_path')
    test_dataloader = DataLoader(dataset=test_dataset, 
                             shuffle=False,
                             batch_size=1, 
                             pin_memory=False, 
                             num_workers=2)
    
    net = FrequencyAwareTransFormer()
    net = net.to(device)
    
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=16)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    last_epoch = 0
    iterations = -1
    best_loss = float("inf")
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')    
        net.load_state_dict(checkpoint)


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        
    if args.test:
        test_epoch(0, test_dataloader, net, criterion,test2=1)
        exit(-1)

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        net.train()
        device = next(net.parameters()).device
        for i, d in enumerate(train_dataloader):
            start_time = time.time()
            d = d.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            out_net = net(d)
            out_criterion = criterion(out_net, d)
            out_criterion["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()
            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i % 2 ==0:
                mse_loss = out_criterion['mse_loss']
                if mse_loss.item() > 0:
                    psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                    psnrs.update(psnr.item())
                else:
                    psnrs.update(100)
                elapsed.update(time.time() - start_time)
                losses.update(out_criterion['loss'].item())
                bpps.update(out_criterion['bpp_loss'].item())
                mse_losses.update(mse_loss.item())
                aux_losses.update(aux_loss.item())
                
            if i % 100 == 0:
                current_time = datetime.now()
                print(    ' | '.join([
                f"{current_time}",
                f'Epoch {epoch}',
                f"{i*len(d)}/{len(train_dataloader.dataset)}",
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'aux_losses Loss {aux_losses.val:.3f} ({aux_losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
           
            ]))
                start_time = time.time()
              
            if (iterations% 5000 == 0 and iterations != 0):
                net.eval()
                loss = test_epoch(iterations, test_dataloader, net, criterion)         
                lr_scheduler.step(loss)
                net.train()
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)               
                if is_best or iterations% 20000 == 0:
                    if args.save :
                        torch.save(net.state_dict, args.save_path)
                    
            iterations = iterations + 1

if __name__ == "__main__":
    main(sys.argv[1:])
