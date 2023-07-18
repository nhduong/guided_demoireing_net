'''
PyTorch source code for "Multiscale Guided Coarse-to-Fine Network for Screenshot Demoiréing"
Project page: https://nhduong.github.io/guided_demoireing_net/
'''

import argparse
import os
import shutil
import math
from enum import Enum
import random
import platform

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted
from glob import glob
import torch.nn.functional as F

import numpy as np
from PIL import Image
from PIL import ImageFile

from tqdm import tqdm
import datetime
import sys

from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from math import log10
import lpips
from math import exp
import cv2

import logging
import traceback
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator

import warnings
warnings.simplefilter('ignore')

def parse():
    parser = argparse.ArgumentParser(description='PyTorch Demoireing Training')

    parser.add_argument('--data_path', default='', type=str,
                        help='data path')
    parser.add_argument('--train_dir', default='', type=str,
                        help='train dir name')
    parser.add_argument('--test_dir', default='', type=str,
                        help='test dir name')
    parser.add_argument('--moire_dir', default='', type=str,
                        help='moire dir name')
    parser.add_argument('--clean_dir', default='', type=str,
                        help='clean dir name')
    parser.add_argument('--data_name', default='', type=str,
                        help='dataset name')
    parser.add_argument('--exp_name', default='spl', type=str,
                        help='experiment name (default: spl)')
    parser.add_argument('--note', default='rev_1', type=str,
                        help='notes (default: rev_1)')
    parser.add_argument('--adaloss', action='store_true',
                        help='Uses adaptive loss.')
    parser.add_argument('--affine', action='store_true',
                        help='Uses affine transformation.')
    parser.add_argument('--l1loss', action='store_true',
                        help='Uses L1 loss.')
    parser.add_argument('--perloss', action='store_true',
                        help='Uses perceptual loss.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size per process (default: 2)')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        metavar='N', help='test mini-batch size per process (default: 2)')
    parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/2: args.lr = args.lr*float(args.batch_size*args.world_size)/2.')
    parser.add_argument('--eta_min', default=0.000001, type=float,
                        metavar='ETA_MIN', help='Learning rate at the end of a cycle.  Will be scaled by <global batch size>/2: args.eta_min = args.eta_min*float(args.batch_size*args.world_size)/2.')
    parser.add_argument('--ada_lamb', default=5.0, type=float,
                        help='ada lamb (default: 5.0)')
    parser.add_argument('--ada_eps', default=1.0, type=float,
                        help='ada eps (default: 1.0)')
    parser.add_argument('--ada_eps_2', default=1.0, type=float,
                        help='ada eps 2 (default: 1.0)')
    parser.add_argument('--num_branches', default=3, type=int,
                        help='number of network branches (default: 3)')
    parser.add_argument('--init_weights', action='store_true',
                        help='initialize weights.')
    parser.add_argument('--T_0', default=50, type=int,
                        metavar='T_0', help='The number of epochs of the first learning cycle (default: 50)')
    parser.add_argument('--print_freq', '-p', default=1000, type=int,
                        metavar='N', help='print frequency (default: 1000)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint .pth.tar')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--calc_mets', action='store_true',
                        help='Calculates metrics during training. Enabling this could slow down the program.')
    parser.add_argument('--calc_val_losses', action='store_true',
                        help='Calculates validation losses during training. Enabling this could slow down the program.')
    parser.add_argument('--calc_train_mets', action='store_true',
                        help='Calculates metrics during training. Enabling this could slow down the program.')
    parser.add_argument('--dont_calc_mets_at_all', action='store_true',
                        help='Does not calculate metrics at all. Enabling this could speed up the program.')
    parser.add_argument('--dont_calc_train_mets', action='store_true',
                        help='Does not calculate metrics during training. Enabling this could speed up the program.')
    parser.add_argument('--log2file', action='store_true',
                        help='export all logs to a file.')
    parser.add_argument('--seed', default=123, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()
    return args


def main():
    # get user inputs
    args = parse()

    # random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = False
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # HunggingFace accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # evaluation metrics
    global best_psnr, cor_ssim, best_epoch
    best_psnr = 0
    cor_ssim = 0
    best_epoch = -1

    # working directory
    time_id = get_current_time()
    log_dir         = os.path.join("outputs", args.exp_name, args.data_name, "[" + args.exp_name + "]_[" + args.data_name + "]_[" + args.note + "]_[" + time_id + "]_[GPU_" + os.environ["CUDA_VISIBLE_DEVICES"] + "]_" + "[" + platform.node() + "]")
    logs_path       = os.path.join(os.path.join(log_dir, "tb"), "train")
    logs_path_val   = os.path.join(os.path.join(log_dir, "tb"), "val")
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(logs_path_val, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "cp"), exist_ok=True)

    # tensorboard logs
    writer = SummaryWriter(log_dir=logs_path, flush_secs=1)
    writer_val = SummaryWriter(log_dir=logs_path_val, flush_secs=1)
    print(">>> Tensorboard logs saved to: {}".format(logs_path))
    print(">>> Tensorboard logs saved to: {}".format(logs_path_val))
    log_fn = os.path.join(log_dir, args.exp_name + "_" + args.data_name + "_" + args.note + "_" + time_id + ".log")
    if args.log2file:
        sys.stdout = open(log_fn, "w")
        sys.stderr = sys.stdout
        logging.basicConfig(filename=log_fn, filemode="a")
        print(">>> Logs saved to: {}".format(log_fn))
    
    writer_val.add_text("start_time", time_id, 0)
    writer_val.flush()

    # starting time
    t0 = datetime.datetime.now()

    # print args to console
    print("===========================")
    print(args)
    print("===========================")
    proc_id = os.getpid()
    print(">>> proccess ID:", proc_id)
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    # adding logs
    args.init_date = "[" + time_id + "]"
    args.proc_id = proc_id
    args.app = "[" + args.exp_name + "_" + args.data_name + "]_[" + args.note + "]"
    args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    args.node = platform.node()
    args.path = log_dir

    # PyTorch device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create model
    print("=> creating model...")
    model = my_model(
        affine=args.affine,
        num_branches=args.num_branches,
    ).to(device)

    # initialize weights
    if args.init_weights:
        model._initialize_weights()
    
    # define loss functions, optimizer, and learning rate scheduler
    criterion_1 = AttL1Loss(lamb=args.ada_lamb, eps=args.ada_eps, eps_2=args.ada_eps_2).to(device)
    criterion_1.print_params()
    criterion_2 = VGGPerceptualLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    print(optimizer)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1, end_factor=args.eta_min / args.lr,
        total_iters=args.T_0 - 1,
    )
    print(scheduler)

    for ggg in optimizer.param_groups:
        ggg['lr'] = args.lr

    # evaluation metrics
    compute_metrics = create_metrics(args, device=device)

    # moire data loaders
    train_dataset = MoireDataset(
        data_path=args.data_path,
        sub_dir=args.train_dir,
        moire_dir=args.moire_dir, clean_dir=args.clean_dir,
        data_name=args.data_name,
        is_training=True,
        transform=None,
        adaloss=args.adaloss,
    )

    val_dataset = MoireDataset(
        data_path=args.data_path,
        sub_dir=args.test_dir,
        moire_dir=args.moire_dir, clean_dir=args.clean_dir,
        data_name=args.data_name,
        is_training=False,
        transform=None,
        adaloss=False, # no adaptive loss for validation
    )

    print("args.batch_size", args.batch_size)
    print("args.test_batch_size", args.test_batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # initialize accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda())
        
        print("=> loading model")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print(">>> scheduler not loaded")
            pass

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print(">>> optimizer not loaded")
            for ggg in optimizer.param_groups:
                ggg['lr'] = scheduler.get_last_lr()[0]
            print(">>> lr reset to {}".format(scheduler.get_last_lr()[0]))
            pass
        
        # optimizer.param_groups[0]['capturable'] = True

        print("=> loaded checkpoint '{}'".format(args.resume))

        if args.evaluate:
            val_loss, val_lossl1, val_lossper, val_psnr, val_ssim, val_lossl1_norm = validate(val_loader, model, device, criterion_1, criterion_2, t0, args.start_epoch - 1, args, writer_val, compute_metrics)
            return
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # start training
    epoch = args.start_epoch
    print(">>> start epoch: {}".format(epoch))

    args.evaluate = args.calc_mets
    print(">>> args.evaluate: {}".format(args.evaluate))

    try:
        while epoch < args.epochs:
            # reset learning rate
            if epoch % args.T_0 == 0: # epoch 50, 100, ...
                for ggg in optimizer.param_groups:
                    ggg['lr'] = args.lr
                
                # reset learning rate scheduler
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1, end_factor=args.eta_min / args.lr,
                    total_iters=args.T_0 - 1,
                )
                print(scheduler)

            lr = optimizer.param_groups[0]["lr"]
            writer_val.add_scalar('z/lr', lr, epoch)
            writer_val.flush()

            # training
            train_loss, train_lossl1, train_lossper, train_psnr, train_ssim, train_lossl1_norm = train(accelerator, train_loader, model, device, criterion_1, criterion_2, optimizer, epoch, args, t0, lr, writer, compute_metrics)

            # tensorboard logs
            writer.add_scalar('loss/loss', train_loss, epoch)
            writer.add_scalar('loss/lossl1', train_lossl1, epoch)
            writer.add_scalar('loss/lossl1_norm', train_lossl1_norm, epoch)
            writer.add_scalar('loss/lossper', train_lossper, epoch)
            if train_psnr > 0 or train_ssim > 0:
                writer.add_scalar('met/psnr', train_psnr, epoch)
                writer.add_scalar('met/ssim', train_ssim, epoch)
            writer.flush()

            # validation
            if not args.dont_calc_mets_at_all:
                if (epoch + 1) % round(args.T_0 / 2) == 0 or epoch % args.T_0 == 0 or args.evaluate:
                    val_loss, val_lossl1, val_lossper, val_psnr, val_ssim, val_lossl1_norm = validate(val_loader, model, device, criterion_1, criterion_2, t0, epoch, args, writer_val, compute_metrics)
            else:
                val_loss = val_lossl1 = val_lossper = val_psnr = val_ssim = val_lossl1_norm = 0

            # tensorboard logs
            if val_loss > 0 or val_lossl1 > 0 or val_lossl1_norm > 0 or val_lossper > 0 or val_psnr > 0 or val_ssim > 0:
                writer_val.add_scalar('loss/loss', val_loss, epoch)
                writer_val.add_scalar('loss/lossl1', val_lossl1, epoch)
                writer_val.add_scalar('loss/lossl1_norm', val_lossl1_norm, epoch)
                writer_val.add_scalar('loss/lossper', val_lossper, epoch)
                writer_val.add_scalar('met/psnr', val_psnr, epoch)
                writer_val.add_scalar('met/ssim', val_ssim, epoch)

            try:
                # model params
                for name, param in model.named_parameters():
                    writer_val.add_histogram("param/" + name, param.clone().cpu().data.numpy(), epoch)
                    if param.grad is not None:
                        writer_val.add_histogram("grad/" + name, param.grad.cpu(), epoch)
            except:
                pass

            writer.flush()
            writer_val.flush()
            
            # learning rate scheduling
            scheduler.step()

            # check if is best
            is_best = val_psnr > best_psnr
            best_psnr = max(val_psnr, best_psnr)

            if is_best:
                best_epoch = epoch
                best_psnr = val_psnr
                cor_ssim = val_ssim
                writer_val.add_text('best', 'best val PSNR {0} | val SSIM {1}\n'.format(best_psnr, cor_ssim), epoch)
                writer_val.flush()

            # save checkpoint
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'cor_ssim': cor_ssim,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
            }, is_best, log_dir, "{:04d}".format(epoch))

            # print to console
            print(
                '## train PSNR {0} | train SSIM {1}\n'
                '>> best val PSNR {2} | val SSIM {3}\n'.format(train_psnr, train_ssim, best_psnr, cor_ssim)
            )

            # next epoch
            epoch += 1

        # training finished!
        writer_val.add_text("finished_time", get_current_time(), 0)
        writer_val.flush()
        writer_val.close()
        writer.close()

        print("finished!")

    except Exception as e:
        logging.error(traceback.format_exc())


def train(accelerator, train_loader, model, device, criterion_1, criterion_2, optimizer, epoch, args, t0, lr, writer, compute_metrics):
    """
    The function `train` trains a model using a given train loader, computes various losses, and updates
    the model parameters using SGD optimization.
    
    :param train_loader: The train_loader is a data loader object that provides batches of training
    data. It is used to iterate over the training dataset during each epoch of training
    :param model: The model is the neural network model that you are training. It takes the moire image
    as input and outputs the predicted clean image
    :param device: The "device" parameter is used to specify the device (CPU or GPU) on which the model
    and data should be loaded. It is typically a torch.device object
    :param criterion_1: The criterion_1 is the loss function used to compute the L1 loss between the
    model's output and the clean image. It is typically a function that takes the output and target
    tensors as inputs and returns the loss value
    :param criterion_2: The `criterion_2` parameter is the loss function used for the perceptual loss.
    It is a function that takes the model output and the ground truth clean image as inputs and returns
    the loss value
    :param optimizer: The optimizer is an object that implements the optimization algorithm. It is used
    to update the model's parameters based on the computed gradients
    :param epoch: The current epoch number of the training process
    :param args: The `args` parameter is a namespace object that contains various arguments and
    hyperparameters for the training process. It is used to configure the behavior of the training loop
    and the model
    :param t0: The variable `t0` is the starting time of the training process. It is used to calculate
    the elapsed time for each epoch
    :param lr: The learning rate used for optimization
    :param writer: The `writer` parameter is an instance of `torch.utils.tensorboard.SummaryWriter`
    which is used to write the training progress and visualizations to TensorBoard
    :param compute_metrics: The `compute_metrics` parameter is a function that is used to compute
    metrics such as PSNR and SSIM. It takes two arguments: the output image and the ground truth image.
    The function should return the computed metrics
    :return: the average values of the losses (total loss, l1 loss, perceptual loss), PSNR, SSIM, and
    normalized l1 loss.
    """

    # training meters
    losses = AverageMeter()
    lossl1 = AverageMeter()
    lossl1_norm = AverageMeter()
    lossper = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    loss_l1_1 = torch.tensor(0.0).to(device)
    norm_loss_l1_1 = torch.tensor(0.0).to(device)
    loss_l1_2 = torch.tensor(0.0).to(device)
    norm_loss_l1_2 = torch.tensor(0.0).to(device)
    loss_l1_3 = torch.tensor(0.0).to(device)
    norm_loss_l1_3 = torch.tensor(0.0).to(device)

    loss_per = torch.tensor(0.0).to(device)

    current_psnr = torch.tensor(0.0).to(device)
    current_ssim = torch.tensor(0.0).to(device)

    # switch to train mode
    model.train()

    # number of samples in the data loader
    train_loader_len = len(train_loader) * args.batch_size
    proc_items = 0
    print(">>> number of train samples: ", train_loader_len)

    with tqdm(total=train_loader_len, dynamic_ncols=True, bar_format="[train] {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}] {desc}", desc="E:%03d | P:%.4f | lr:%.6e | L:%.2f | L1:%.2f | L1n:%.2f | Lp:%.2f | S:%.4f (%s)" % (epoch, psnr.avg, optimizer.param_groups[0]["lr"], losses.avg, lossl1.avg, lossl1_norm.avg, lossper.avg, ssim.avg, str(datetime.datetime.now() - t0).split(".")[0]), ascii="-#") as pbar:
        for i_batch, data in enumerate(train_loader):
            # move data to the same device as model
            moire_img = data["moire"].to(device)
            clean_img = data["clean"].to(device)
            clean_freq = data["clean_freq"].to(device)

            # compute pixel rarity (based on "histogram")
            if clean_freq.shape[1] == 1:
                clean_freq = None
            else:
                clean_freq = clean_freq.long()
                for ib in range(clean_img.shape[0]):
                    freq_its = clean_freq[ib]
                    bins = torch.bincount(freq_its)
                    clean_freq[ib] = bins[freq_its.long()]
                clean_freq = clean_freq.reshape(clean_img.shape[0], 1, clean_img.shape[2], clean_img.shape[3])
                clean_freq = clean_freq.float()
                clean_freq /= clean_img.shape[2] * clean_img.shape[3]

                # blur
                kersize = int(min(clean_img.shape[2], clean_img.shape[3]) / 68)
                if kersize % 2 == 0:
                    kersize += 1
                clean_freq = transforms.GaussianBlur(kernel_size=(kersize, kersize), sigma=(5.0, 5.0))(clean_freq)

            # multi-scale ground truths
            clean_img_2 = F.interpolate(clean_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            clean_img_3 = F.interpolate(clean_img, scale_factor=0.25, mode='bilinear', align_corners=False)
            if args.num_branches >= 4:
                clean_img_4 = F.interpolate(clean_img, scale_factor=0.125, mode='bilinear', align_corners=False)
            if args.num_branches >= 5:
                clean_img_5 = F.interpolate(clean_img, scale_factor=0.0625, mode='bilinear', align_corners=False)

            try:
                clean_freq_2 = F.interpolate(clean_freq, scale_factor=0.5, mode='bilinear', align_corners=False)
                clean_freq_3 = F.interpolate(clean_freq, scale_factor=0.25, mode='bilinear', align_corners=False)
                if args.num_branches >= 4:
                    clean_freq_4 = F.interpolate(clean_freq, scale_factor=0.125, mode='bilinear', align_corners=False)
                if args.num_branches >= 5:
                    clean_freq_5 = F.interpolate(clean_freq, scale_factor=0.0625, mode='bilinear', align_corners=False)
            except:
                clean_freq_2 = None
                clean_freq_3 = None
                if args.num_branches >= 4:
                    clean_freq_4 = None
                if args.num_branches >= 5:
                    clean_freq_5 = None

            # compute outputs
            if args.num_branches == 3:
                output, output_2, output_3 = model(moire_img)
            elif args.num_branches == 4:
                output, output_2, output_3, output_4 = model(moire_img)
            elif args.num_branches == 5:
                output, output_2, output_3, output_4, output_5 = model(moire_img)

            # l1 norm
            if args.l1loss:
                loss_l1_1, norm_loss_l1_1 = criterion_1(output, clean_img, clean_freq, max_lr=args.lr, min_lr=args.eta_min, lr=lr)
                loss_l1_2, norm_loss_l1_2 = criterion_1(output_2, clean_img_2, clean_freq_2, max_lr=args.lr, min_lr=args.eta_min, lr=lr)
                loss_l1_3, norm_loss_l1_3 = criterion_1(output_3, clean_img_3, clean_freq_3, max_lr=args.lr, min_lr=args.eta_min, lr=lr)
                if args.num_branches >= 4:
                    loss_l1_4, norm_loss_l1_4 = criterion_1(output_4, clean_img_4, clean_freq_4, max_lr=args.lr, min_lr=args.eta_min, lr=lr)
                if args.num_branches >= 5:
                    loss_l1_5, norm_loss_l1_5 = criterion_1(output_5, clean_img_5, clean_freq_5, max_lr=args.lr, min_lr=args.eta_min, lr=lr)

            loss_l1 = loss_l1_1 + loss_l1_2 + loss_l1_3
            norm_loss_l1 = norm_loss_l1_1 + norm_loss_l1_2 + norm_loss_l1_3
            if args.num_branches >= 4:
                loss_l1 += loss_l1_4
                norm_loss_l1 += norm_loss_l1_4
            if args.num_branches >= 5:
                loss_l1 += loss_l1_5
                norm_loss_l1 += norm_loss_l1_5

            # perceptual loss
            if args.perloss:
                loss_per = criterion_2(output, clean_img, feature_layers=[2]) + \
                    criterion_2(output_2, clean_img_2, feature_layers=[2]) + \
                        criterion_2(output_3, clean_img_3, feature_layers=[2])
                if args.num_branches >= 4:
                    loss_per += criterion_2(output_4, clean_img_4, feature_layers=[2])
                if args.num_branches >= 5:
                    loss_per += criterion_2(output_5, clean_img_5, feature_layers=[2])

            # total loss
            loss = loss_l1 + loss_per

            # compute gradient and do SGD step
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # measure PSNR and SSIM
            if not args.dont_calc_mets_at_all and args.calc_train_mets and not args.dont_calc_train_mets:
                if (epoch + 1) % round(args.T_0 / 2) == 0 or epoch % args.T_0 == 0 or args.evaluate:
                    with torch.no_grad():
                        try:
                            # _, current_psnr, current_ssim = compute_metrics.compute(torch.clamp(output, 0, 1), clean_img)
                            _, current_psnr, current_ssim = compute_metrics.compute(output, clean_img)
                        except Exception as e:
                            pass

            # update meters
            losses.update(loss.item(), moire_img.size(0))
            lossl1.update(loss_l1.item(), moire_img.size(0))
            lossl1_norm.update(norm_loss_l1.item(), moire_img.size(0))
            lossper.update(loss_per.item(), moire_img.size(0))
            try:
                psnr.update(current_psnr.item(), moire_img.size(0))
                ssim.update(current_ssim.item(), moire_img.size(0))
            except:
                psnr.update(current_psnr, moire_img.size(0))
                ssim.update(current_ssim, moire_img.size(0))

            # update progress bar
            proc_items += moire_img.size(0)
            if train_loader_len > 0:
                if (i_batch == train_loader_len - 1) or (i_batch % max(5, round(train_loader_len / args.print_freq)) == 0):
                    pbar.set_description_str("E:%03d | P:%.4f | lr:%.6e | L:%.2f | L1:%.2f | L1n:%.2f | Lp:%.2f | S:%.4f (%s)" % (epoch, psnr.avg, optimizer.param_groups[0]["lr"], losses.avg, lossl1.avg, lossl1_norm.avg, lossper.avg, ssim.avg, str(datetime.datetime.now() - t0).split(".")[0]), refresh=False)
                    pbar.update(proc_items)
                    proc_items = 0

    # tensorboard images
    with torch.no_grad():
        num_dis_imgs = min(6, args.batch_size)
        if args.num_branches == 3:
            num_imgs_per_row = 5
            dis_text = "0/in_gt_out1_out2_out3"
        elif args.num_branches == 4:
            num_imgs_per_row = 6
            dis_text = "0/in_gt_out1_out2_out3_out4"
        elif args.num_branches == 5:
            num_imgs_per_row = 7
            dis_text = "0/in_gt_out1_out2_out3_out4_out5"

        # re-scale outputs to save space
        output_2 = F.interpolate(output_2, size=(output.shape[2], output.shape[3]), mode="bilinear", align_corners=False)
        output_3 = F.interpolate(output_3, size=(output.shape[2], output.shape[3]), mode="bilinear", align_corners=False)
        if args.num_branches >= 4:
            output_4 = F.interpolate(output_4, size=(output.shape[2], output.shape[3]), mode="bilinear", align_corners=False)
        if args.num_branches >= 5:
            output_5 = F.interpolate(output_5, size=(output.shape[2], output.shape[3]), mode="bilinear", align_corners=False)

        # add images to tensorboard
        imgs = []
        for i_id in np.arange(0, num_dis_imgs):
            imgs.append(moire_img[i_id])
            imgs.append(torch.clamp(output[i_id], 0, 1))
            imgs.append(torch.clamp(output_2[i_id], 0, 1))
            imgs.append(torch.clamp(output_3[i_id], 0, 1))
            if args.num_branches >= 4:
                imgs.append(torch.clamp(output_4[i_id], 0, 1))
            if args.num_branches >= 5:
                imgs.append(torch.clamp(output_5[i_id], 0, 1))
            imgs.append(clean_img[i_id])
        
        grid = torchvision.utils.make_grid(imgs, num_imgs_per_row)
        writer.add_image(dis_text, grid, epoch)
        writer.flush()

    return losses.avg, lossl1.avg, lossper.avg, psnr.avg, ssim.avg, lossl1_norm.avg


def validate(val_loader, model, device, criterion_1, criterion_2, t0, epoch, args, writer, compute_metrics):    
    """
    The function `validate` is used to evaluate the performance of a model on a validation dataset,
    calculating various metrics such as loss, PSNR, and SSIM, and visualizing the input, ground truth,
    and output images using TensorBoard.
    
    :param val_loader: The validation data loader, which provides batches of data for validation
    :param model: The `model` parameter is the neural network model that will be used for validation. It
    should be an instance of a PyTorch model class
    :param device: The "device" parameter is used to specify the device (CPU or GPU) on which the model
    and data should be loaded. It is typically a torch.device object
    :param criterion_1: The `criterion_1` parameter is the loss function used to compute the L1 loss
    between the model's output and the clean image. It is typically a function that takes the output and
    target tensors as inputs and returns the loss value
    :param criterion_2: The `criterion_2` parameter is the loss function used to calculate the
    perceptual loss. It is a function that takes the output of the model and the clean image as inputs
    and returns the loss value
    :param t0: The parameter `t0` is not explicitly defined in the code snippet you provided. It is
    likely defined elsewhere in your code. Please provide more information or the definition of `t0` for
    further assistance
    :param epoch: The current epoch number
    :param args: The `args` parameter is a dictionary or object that contains various configuration
    settings for the validation process. It likely includes settings such as the batch size, number of
    branches, data name, whether to use L1 loss or perceptual loss, and other parameters specific to the
    model being used
    :param writer: The `writer` parameter is an instance of `torch.utils.tensorboard.SummaryWriter`
    which is used to write the training and validation metrics to TensorBoard. It is used to visualize
    the training progress and monitor the performance of the model
    :param compute_metrics: The `compute_metrics` parameter is a function that is used to compute the
    metrics (PSNR and SSIM) between the output of the model and the clean image. It takes two arguments:
    the output tensor and the clean image tensor. The function should return the metrics as
    floating-point numbers
    :return: the average values of the losses (total loss, l1 loss, perceptual loss), PSNR (Peak
    Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and normalized l1 loss.
    """

    # validation meters
    losses = AverageMeter()
    lossl1 = AverageMeter()
    lossl1_norm = AverageMeter()
    lossper = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()

    loss_l1_1 = torch.tensor(0.0).to(device)
    norm_loss_l1_1 = torch.tensor(0.0).to(device)

    loss_per = torch.tensor(0.0).to(device)

    current_psnr = torch.tensor(0.0).to(device)
    current_ssim = torch.tensor(0.0).to(device)

    # number of samples in the data loader
    proc_items = 0
    val_loader_len = len(val_loader) * args.test_batch_size
    print(">>> number of val samples: ", val_loader_len)

    # random image id for tensorboard
    ran_img_id = random.randint(0, len(val_loader) - 1)
    print(">>> ran_img_id: ", ran_img_id)

    val_moire_img = None
    val_clean_img = None
    val_output = None

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(total=val_loader_len, dynamic_ncols=True, bar_format="> [val] {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}] {desc}", desc="E:%03d | P:%.4f | L:%.2f | L1:%.2f | L1n:%.2f | Lp:%.2f | S:%.4f (%s)" % (epoch, psnr.avg, losses.avg, lossl1.avg, lossl1_norm.avg, lossper.avg, ssim.avg, str(datetime.datetime.now() - t0).split(".")[0]), ascii="-#") as pbar:
            for i_batch, data in enumerate(val_loader):
                # move data to the same device as model
                moire_img = data["moire"].to(device)
                clean_img = data["clean"].to(device)

                # image padding for multi-scale inference
                _, _, h, w = moire_img.size()
                if args.data_name == "fhdmi":
                    if args.num_branches == 3:
                        w_pad = (math.ceil(w/32)*32 - w) // 2
                        h_pad = (math.ceil(h/32)*32 - h) // 2
                    elif args.num_branches == 4:
                        w_pad = (math.ceil(w/64)*64 - w) // 2
                        h_pad = (math.ceil(h/64)*64 - h) // 2
                    elif args.num_branches == 5:
                        w_pad = (math.ceil(w/128)*128 - w) // 2
                        h_pad = (math.ceil(h/128)*128 - h) // 2
                else:
                    w_pad = (math.ceil(w/32)*32 - w) // 2
                    h_pad = (math.ceil(h/32)*32 - h) // 2
                
                moire_img = img_pad(moire_img, w_r=w_pad, h_r=h_pad)

                # compute output
                if args.num_branches == 3:
                    output, _, _ = model(moire_img)
                elif args.num_branches == 4:
                    output, _, _, _ = model(moire_img)
                elif args.num_branches == 5:
                    output, _, _, _, _ = model(moire_img)

                # remove padding
                if h_pad != 0:
                    output = output[:, :, h_pad:-h_pad, :]
                if w_pad != 0:
                    output = output[:, :, :, w_pad:-w_pad]

                # l1 norm
                if args.l1loss and args.calc_val_losses:
                    loss_l1_1, norm_loss_l1_1 = criterion_1(output, clean_img)

                loss_l1 = loss_l1_1
                norm_loss_l1 = norm_loss_l1_1
                
                # perceptual loss
                if args.perloss and args.calc_val_losses:
                    loss_per = criterion_2(output, clean_img, feature_layers=[2])

                # total loss
                loss = loss_l1 + loss_per

                # save images for tensorboard visualization
                if i_batch == ran_img_id:
                    val_moire_img = moire_img.detach().cpu()
                    val_clean_img = clean_img.detach().cpu()
                    val_output = output.detach().cpu()
                
                # measure PSNR and SSIM
                if not args.dont_calc_mets_at_all:
                    if (epoch + 1) % round(args.T_0 / 2) == 0 or epoch % args.T_0 == 0 or args.evaluate:
                        try:
                            _, current_psnr, current_ssim = compute_metrics.compute(output, clean_img)
                        except Exception as e:
                            pass

                # update meters
                losses.update(loss, moire_img.size(0))
                lossl1.update(loss_l1, moire_img.size(0))
                lossl1_norm.update(norm_loss_l1, moire_img.size(0))
                lossper.update(loss_per, moire_img.size(0))
                psnr.update(current_psnr, moire_img.size(0))
                ssim.update(current_ssim, moire_img.size(0))

                # update progress bar
                proc_items += moire_img.size(0)

                if val_loader_len > 0:
                    if (i_batch == val_loader_len - 1) or (i_batch % max(5, round(val_loader_len / args.print_freq)) == 0):
                        pbar.set_description_str("E:%03d | P:%.4f | L:%.2f | L1:%.2f | L1n:%.2f | Lp:%.2f | S:%.4f (%s)" % (epoch, psnr.avg, losses.avg, lossl1.avg, lossl1_norm.avg, lossper.avg, ssim.avg, str(datetime.datetime.now() - t0).split(".")[0]), refresh=False)
                        pbar.update(proc_items)
                        proc_items = 0

                # if i_batch > 1:
                #     break

    print('## val PSNR {psnr.avg:.5f} | val SSIM {ssim.avg:.5f}'.format(psnr=psnr, ssim=ssim))

    # tensorboard images
    try:
        num_dis_imgs = min(6, args.test_batch_size)
        _, _, img_h, img_w = val_moire_img.shape

        # resize images to save space
        val_clean_img = F.interpolate(val_clean_img, size=(img_h, img_w), mode="bilinear", align_corners=False)
        val_output = F.interpolate(val_output, size=(img_h, img_w), mode="bilinear", align_corners=False)

        imgs = torch.cat(
            (
                val_moire_img[:num_dis_imgs],
                val_clean_img[:num_dis_imgs],
                torch.clamp(val_output[:num_dis_imgs], 0, 1),
            ),
            dim=0
        )
        grid = torchvision.utils.make_grid(imgs, num_dis_imgs)
        writer.add_image("in_gt_out", grid, epoch)
        writer.flush()
    except:
        logging.warning(traceback.format_exc())
        pass

    return losses.avg, lossl1.avg, lossper.avg, psnr.avg, ssim.avg, lossl1_norm.avg


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    """
    Fast pytorch implementation for SSIM, referred from
    "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py"
    """
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class PSNR(torch.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        psnr = -10*torch.log10(torch.mean((img1-img2)**2))
        
        return psnr


def generate_1d_gaussian_kernel():
    return cv2.getGaussianKernel(11, 1.5)


def generate_2d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    return np.outer(kernel, kernel.transpose())


def generate_3d_gaussian_kernel():
    kernel = generate_1d_gaussian_kernel()
    window = generate_2d_gaussian_kernel()
    return np.stack([window * k for k in kernel], axis=0)


class MATLAB_SSIM(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(MATLAB_SSIM, self).__init__()
        self.device = device
        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = torch.tensor(generate_3d_gaussian_kernel())
        self.conv3d = conv3d.to(device)

        conv2d = torch.nn.Conv2d(1, 1, (11, 11), stride=1, padding=(5, 5), bias=False, padding_mode='replicate')
        conv2d.weight.requires_grad = False
        conv2d.weight[0, 0, :, :] = torch.tensor(generate_2d_gaussian_kernel())
        self.conv2d = conv2d.to(device)

    def forward(self, img1, img2):
        assert len(img1.shape) == len(img2.shape)
        with torch.no_grad():
            img1 = torch.tensor(img1).to(self.device).float()
            img2 = torch.tensor(img2).to(self.device).float()

            if len(img1.shape) == 2:
                conv = self.conv2d
            elif len(img1.shape) == 3:
                conv = self.conv3d
            else:
                raise not NotImplementedError('only support 2d / 3d images.')
            return self._ssim(img1, img2, conv)

    def _ssim(self, img1, img2, conv):
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = conv(img1)
        mu2 = conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv(img1 ** 2) - mu1_sq
        sigma2_sq = conv(img2 ** 2) - mu2_sq
        sigma12 = conv(img1 * img2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) *
                    (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                           (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = torchvision.utils.make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    
    
    return img_np.astype(out_type)


class create_metrics():
    """
       https://github.com/CVMI-Lab/UHDM
    """
    def __init__(self, args, device):
        self.data_type = args.data_name
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        self.fast_ssim = SSIM()
        self.fast_psnr = PSNR()
        self.matlab_ssim = MATLAB_SSIM(device=device)
        self.args = args
        self.device = device

    def compute(self, out_img, gt):
        if self.data_type == 'uhdm':
            res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt, self.args)
        elif self.data_type == 'fhdmi':
            res_psnr, res_ssim = self.skimage_psnr_ssim(out_img, gt, self.args)
        elif self.data_type == 'tip18':
            res_psnr, res_ssim = self.matlab_psnr_ssim(out_img, gt, self.args)
        elif self.data_type == 'aim':
            res_psnr, res_ssim = self.aim_psnr_ssim(out_img, gt, self.args)
        else:
            print('Unrecognized data_type for evaluation!')
            raise NotImplementedError

        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)

        # calculate LPIPS
        if pre.shape[0] == 1:
            res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).item()
        else:
            res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).mean().item()

        # res_lpips = torch.tensor(0.0).to(self.device)
        return res_lpips, res_psnr, res_ssim


    def fast_psnr_ssim(self, out_img, gt, args):
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)
        psnr = self.fast_psnr(pre, tar)
        ssim = self.fast_ssim(pre, tar)
        return psnr, ssim

    def skimage_psnr_ssim(self, out_img, gt, args):
        """
        Same with the previous SOTA FHDe2Net: https://github.com/PKU-IMRE/FHDe2Net/blob/main/test.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        # if args.calc_ssim:
        #     ssim = ski_ssim(mt1, mi1, multichannel=True)
        # else:
        #     ssim = torch.tensor(0.0).to(device)
        # print(ssim)
        return psnr, ssim

    def matlab_psnr_ssim(self, out_img, gt, args):
        """
        A pytorch implementation for reproducing SSIM results when using MATLAB
        same with the previous SOTA MopNet: https://github.com/PKU-IMRE/MopNet/blob/master/test_with_matlabcode.m
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = self.matlab_ssim(mt1, mi1)
        return psnr, ssim

    def aim_psnr_ssim(self, out_img, gt, args):
        """
        Same with the previous SOTA MBCNN: https://github.com/zhenngbolun/Learnbale_Bandpass_Filter/blob/master/main_multiscale.py
        """
        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        mi1 = mi1.astype(np.float32) / 255.0
        mt1 = mt1.astype(np.float32) / 255.0
        psnr = 10 * log10(1 / np.mean((mt1 - mi1) ** 2))
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        return psnr, ssim


def save_checkpoint(state, is_best, log_dir, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(log_dir, "cp", epoch + "_" + filename))
    if is_best:
        shutil.copyfile(os.path.join(log_dir, "cp", epoch + "_" + filename), os.path.join(log_dir, "cp", epoch + '_model_best.pth.tar'))
    return os.path.join(log_dir, "cp", epoch + "_" + filename), epoch


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt=':f', summary_type=Summary.AVERAGE):
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# The `AttL1Loss` class is a custom loss function that calculates the L1 loss between the output and
# clean image, with an additional attention mechanism based on frequency information and pixel difference.
# Please refer to our paper "Multiscale Guided Coarse-to-Fine Network for Screenshot Demoiréing" for more details.
class AttL1Loss(nn.Module):
    def __init__(self, lamb=5.0, eps=1, eps_2=1):
        super(AttL1Loss, self).__init__()
        self.lamb = lamb
        self.eps = eps
        self.eps_2 = eps_2

    def print_params(self):
        print("lambda: %.10f, epsilon: %.10f, epsilon_2: %.10f" % (self.lamb, self.eps, self.eps_2))
 
    def forward(self, output, clean_img, clean_img_freq=None, calc_mean=True, eps=None, eps_2=None, lamb=None, max_lr=0.0002, min_lr=0.000005, lr=0.0002):
        l1 = torch.abs(output - clean_img)

        if clean_img_freq is not None and eps is not None and eps_2 is not None and lamb is not None:
            inv_freqs = 1.0 - eps_2*clean_img_freq + eps
            pix_diff = torch.pow(lamb, l1)
            att_loss = torch.pow((inv_freqs * pix_diff), math.log(max_lr/lr, max_lr/min_lr)) * l1
        elif clean_img_freq is not None and self.eps is not None and self.eps_2 is not None and self.lamb is not None:
            inv_freqs = 1.0 - self.eps_2*clean_img_freq + self.eps
            pix_diff = torch.pow(self.lamb, l1)
            att_loss = torch.pow((inv_freqs * pix_diff), math.log(max_lr/lr, max_lr/min_lr)) * l1
        else:
            # Just an l1 norm
            att_loss = l1
        
        if calc_mean:
            return torch.mean(att_loss), torch.mean(l1)
        else:
            return att_loss, l1


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, l1_loss=None, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.l1_loss = l1_loss

    def forward(self, moire_img, clean_img, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if moire_img.shape[1] != 3:
            moire_img = moire_img.repeat(1, 3, 1, 1)
            clean_img = clean_img.repeat(1, 3, 1, 1)
        moire_img = (moire_img-self.mean) / self.std
        clean_img = (clean_img-self.mean) / self.std
        if self.resize:
            moire_img = self.transform(moire_img, mode='bilinear', size=(224, 224), align_corners=False)
            clean_img = self.transform(clean_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = moire_img
        y = clean_img
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if self.l1_loss is None:
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += self.l1_loss(
                        output=x, clean_img=y,
                        clean_img_freq=0, calc_mean=True, eps=0, eps_2=0,
                    )
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def img_pad(x, h_r=0, w_r=0):
    '''
    Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
    the mean values are calculated from UHDM training set, yielding similar performance.
    '''
    x1 = F.pad(x[:, 0:1, ...], (w_r, w_r, h_r, h_r), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_r, w_r, h_r, h_r), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_r, w_r, h_r, h_r), value=0.3912)
    # x1 = F.pad(x[:,0:1,...], (w_r,w_r,h_r,h_r), value = 0.5165)
    # x2 = F.pad(x[:,1:2,...], (w_r,w_r,h_r,h_r), value = 0.4952)
    # x3 = F.pad(x[:,2:3,...], (w_r,w_r,h_r,h_r), value = 0.4695)
    y = torch.cat([x1, x2, x3], dim=1)

    return y


# The guided coarse-to-fine network for screenshot demoiréing
class my_model(nn.Module):
    def __init__(self,
                 feat_num=48,
                 inter_feat_num=32,
                 affine=True,
                 num_branches=5,
                 ):
        super(my_model, self).__init__()
        self.affine = affine
        self.num_branches = num_branches

        self.conv_first = nn.Sequential(
            nn.Conv2d(12, 3 * feat_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.ddb_first = DDB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)

        # down 1/2
        self.down_1 = nn.Sequential(
            nn.Conv2d(3 * feat_num, 3 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        # down 1/4
        self.down_2 = nn.Sequential(
            nn.Conv2d(3 * feat_num, 3 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        if self.num_branches >= 4:
            # down 1/8
            self.down_3 = nn.Sequential(
                nn.Conv2d(3 * feat_num, 3 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

        if self.num_branches >= 5:
            # down 1/16
            self.down_4 = nn.Sequential(
                nn.Conv2d(3 * feat_num, 3 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )

        # share compression
        self.ddb = DDB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)
        self.mgrb_block_1 = MGRB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=affine)
        self.mgrb_block_2 = MGRB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=affine)
        self.mgrb_block_3 = MGRB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=affine)

        if self.num_branches >= 5:
            # after compression 1/16
            self.preconv_4 = conv_relu(3 * feat_num, 3 * feat_num, 3, padding=1)
            self.conv_4 = conv(in_channel=3 * feat_num, out_channel=12, kernel_size=3, padding=1)

            # combine 1/16 and 1/8
            self.ftb_3 = FTB(3 * feat_num, 3 * feat_num)
            self.fusion_3 = CAB(6 * feat_num)

        if self.num_branches >= 4:
            # after compression 1/8
            self.preconv_3 = conv_relu(3 * feat_num, 3 * feat_num, 3, padding=1)
            self.conv_3 = conv(in_channel=3 * feat_num, out_channel=12, kernel_size=3, padding=1)

            # combine 1/8 and 1/4
            self.ftb_2 = FTB(3 * feat_num, 3 * feat_num)
            self.fusion_2 = CAB(6 * feat_num)

        # after compression 1/4
        self.preconv_2 = conv_relu(3 * feat_num, 3 * feat_num, 3, padding=1)
        self.conv_2 = conv(in_channel=3 * feat_num, out_channel=12, kernel_size=3, padding=1)

        # combine 1/4 and 1/2
        self.ftb_1 = FTB(3 * feat_num, 3 * feat_num)
        self.fusion_1 = CAB(6 * feat_num)

        # after compression 1/2
        self.preconv_1 = conv_relu(3 * feat_num, 3 * feat_num, 3, padding=1)
        self.conv_1 = conv(in_channel=3 * feat_num, out_channel=12, kernel_size=3, padding=1)

        # combine 1/2 and 1
        self.ftb_0 = FTB(3 * feat_num, 3 * feat_num)
        self.fusion_0 = CAB(6 * feat_num)

        # after compression 1
        self.conv_0 = conv(in_channel=3 * feat_num, out_channel=12, kernel_size=3, padding=1)
            
    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)
        x = self.ddb_first(x) 
       
        # down 1/2
        x_down_1 = self.down_1(x)
        # down 1/4
        x_down_2 = self.down_2(x_down_1)

        if self.num_branches >= 4:
            # down 1/8
            x_down_3 = self.down_3(x_down_2)

        if self.num_branches >= 5:
            # down 1/16
            x_down_4 = self.down_4(x_down_3)

        if self.num_branches >= 5:
            # share compression 1/16
            x_down_4 = self.ddb(x_down_4)
            x_down_4_out = self.mgrb_block_1(x_down_4)
            x_down_4_out = self.mgrb_block_2(x_down_4_out)
            x_down_4_out = self.mgrb_block_3(x_down_4_out)

            x_down_4_out_feat = self.preconv_4(x_down_4_out)
            x_down_4_out_image = self.conv_4(x_down_4_out)
            x_down_4_out_image = F.pixel_shuffle(x_down_4_out_image, 2)
            x_down_4_out_feat = F.interpolate(x_down_4_out_feat, scale_factor=2, mode='bilinear')

            # combine 1/16 and 1/8
            x_down_3 = self.ftb_3((x_down_3, x_down_4_out_feat), self.affine)
            x_down_3 = self.fusion_3(x_down_3, x_down_4_out_feat)

        if self.num_branches >= 4:
            # share compression 1/8
            x_down_3 = self.ddb(x_down_3)
            x_down_3_out = self.mgrb_block_1(x_down_3)
            x_down_3_out = self.mgrb_block_2(x_down_3_out)
            x_down_3_out = self.mgrb_block_3(x_down_3_out)

            x_down_3_out_feat = self.preconv_3(x_down_3_out)
            x_down_3_out_image = self.conv_3(x_down_3_out)
            x_down_3_out_image = F.pixel_shuffle(x_down_3_out_image, 2)
            x_down_3_out_feat = F.interpolate(x_down_3_out_feat, scale_factor=2, mode='bilinear')

            # combine 1/8 and 1/4
            x_down_2 = self.ftb_2((x_down_2, x_down_3_out_feat), self.affine)
            x_down_2 = self.fusion_2(x_down_2, x_down_3_out_feat)

        # share compression 1/4
        x_down_2 = self.ddb(x_down_2)
        x_down_2_out = self.mgrb_block_1(x_down_2)
        x_down_2_out = self.mgrb_block_2(x_down_2_out)
        x_down_2_out = self.mgrb_block_3(x_down_2_out)
        
        x_down_2_out_feat = self.preconv_2(x_down_2_out)
        x_down_2_out_image = self.conv_2(x_down_2_out)
        x_down_2_out_image = F.pixel_shuffle(x_down_2_out_image, 2)
        x_down_2_out_feat = F.interpolate(x_down_2_out_feat, scale_factor=2, mode='bilinear')
    
        # combine 1/4 and 1/2
        x_down_1 = self.ftb_1((x_down_1, x_down_2_out_feat), self.affine)
        x_down_1 = self.fusion_1(x_down_1, x_down_2_out_feat)
        
        # share compression 1/2
        x_down_1 = self.ddb(x_down_1)
        x_down_1_out = self.mgrb_block_1(x_down_1)
        x_down_1_out = self.mgrb_block_2(x_down_1_out)
        x_down_1_out = self.mgrb_block_3(x_down_1_out)

        x_down_1_out_feat = self.preconv_1(x_down_1_out)
        x_down_1_out_image = self.conv_1(x_down_1_out)
        x_down_1_out_image = F.pixel_shuffle(x_down_1_out_image, 2)
        x_down_1_out_feat = F.interpolate(x_down_1_out_feat, scale_factor=2, mode='bilinear')
        
        # combine 1/2 and 1
        x = self.ftb_0((x, x_down_1_out_feat), self.affine)
        x = self.fusion_0(x, x_down_1_out_feat)

        # share compression 1
        x = self.ddb(x)
        x_out = self.mgrb_block_1(x)
        x_out = self.mgrb_block_2(x_out)
        x_out = self.mgrb_block_3(x_out)

        x_out = self.conv_0(x_out)
        x_out = F.pixel_shuffle(x_out, 2)

        if self.num_branches == 5:
            return x_out, x_down_1_out_image, x_down_2_out_image, x_down_3_out_image, x_down_4_out_image
        elif self.num_branches == 4:
            return x_out, x_down_1_out_image, x_down_2_out_image, x_down_3_out_image
        else:
            return x_out, x_down_1_out_image, x_down_2_out_image

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class FTB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FTB, self).__init__()
        self.FTB_scale_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_scale_conv1 = nn.Conv2d(in_channel, out_channel, 1)
        self.FTB_shift_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_shift_conv1 = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x, affine=True, skip=False):
        if affine:
            # x[0]: fea; x[1]: cond
            scale = self.FTB_scale_conv1(F.leaky_relu(self.FTB_scale_conv0(x[1]), 0.1, inplace=True))
            shift = self.FTB_shift_conv1(F.leaky_relu(self.FTB_shift_conv0(x[1]), 0.1, inplace=True))
            if skip:
                return x[0] * (scale + 1) + shift + x[0]  # + x[0] in case x[1] is clean (=0)
            else:
                return x[0] * (scale + 1) + shift
        else:
            return x[0]


class MGRB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num, affine):
        super(MGRB, self).__init__()

        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion_2 = CAB(2 * in_channel)
        self.ftb_2 = FTB(in_channel, in_channel)
        self.fusion_4 = CAB(2 * in_channel)
        self.ftb_4 = FTB(in_channel, in_channel)

        # !!! wider model with mgrb inside mgrb

        self.affine = affine

    def forward(self, x):
        x_0 = x

        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y_4 = self.basic_block_4(x_4)
        y_4 = F.interpolate(y_4, scale_factor=2, mode='bilinear')

        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        y_2 = self.basic_block_2(self.fusion_4(self.ftb_4((x_2, y_4), self.affine), y_4))
        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')

        y_0 = self.basic_block(self.fusion_2(self.ftb_2((x_0, y_2), self.affine), y_2))

        y = x + y_0

        return y


class CAB(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CAB, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out = torch.cat([out0, out2], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)
        w0, w2 = torch.chunk(out, 2, dim=1)
        x = x0 * w0 + x2 * w2

        return x


class DDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


def get_current_time():
    datetime_now = str(datetime.datetime.now())
    datetime_now = datetime_now.replace(":", ".")
    # datetime_now = datetime_now.replace("-", ".")
    datetime_now = datetime_now.replace(" ", "]_[")
    return datetime_now


class MoireDataset(Dataset):
    """Moire dataset."""
    def __init__(self, data_path, sub_dir, moire_dir, clean_dir, data_name, is_training, transform=None, adaloss=False):
        """
        The function initializes a dataset object with a given data path, subdirectory, moire directory,
        clean directory, data name, training flag, transformation function, and adaloss flag.
        
        :param data_path: The `data_path` parameter is the path to the directory where the data is
        stored
        :param sub_dir: The `sub_dir` parameter is a string that represents the subdirectory within the
        `data_path` directory where the moire and clean images are located
        :param moire_dir: The `moire_dir` parameter is the directory where the moire images are stored
        :param clean_dir: The `clean_dir` parameter is the directory where the clean images are stored
        :param data_name: The `data_name` parameter is a string that specifies the name of the dataset
        being used. It is used to determine the file paths for the moire and clean images. If the
        `data_name` is "uhdm", the moire and clean images are located in the same directory. Otherwise
        :param is_training: The `is_training` parameter is a boolean value that indicates whether the
        dataset is being used for training or not. If `is_training` is `True`, it means that the dataset
        is being used for training. If `is_training` is `False`, it means that the dataset is being used
        :param transform: The `transform` parameter is an optional argument that allows you to specify a
        transformation or a series of transformations to be applied to the data. These transformations
        can include resizing, cropping, rotating, flipping, normalizing, etc. It is commonly used to
        preprocess the data before feeding it into a machine learning
        :param adaloss: The `adaloss` parameter is a boolean flag that indicates whether to use the
        AdaLoss algorithm. AdaLoss is a loss function that adapts the loss weights based on the
        difficulty of each training sample. If `adaloss` is set to `True`, the AdaLoss algorithm will be
        used during, defaults to False (optional)
        """
        if data_name == "uhdm":
            if not is_training:
                self.moire_list = natsorted(glob(os.path.join(data_path, sub_dir, "*_moire.jpg")))
                self.clean_list = natsorted(glob(os.path.join(data_path, sub_dir, "*_gt.jpg")))
            else:
                self.moire_list = []
                self.clean_list = []
                for dir_path in natsorted(glob(os.path.join(data_path, sub_dir, "*"))):
                    for moire_fn in natsorted(glob(os.path.join(data_path, sub_dir, dir_path, "*_moire.jpg"))):
                        self.moire_list.append(moire_fn)
                        self.clean_list.append(moire_fn.replace("_moire.jpg", "_gt.jpg"))
        else:
            self.moire_list = natsorted(glob(os.path.join(data_path, sub_dir, moire_dir, "*")))
            self.clean_list = natsorted(glob(os.path.join(data_path, sub_dir, clean_dir, "*")))

        self.data_name = data_name
        self.is_training = is_training
        self.transform = transform
        self.adaloss = adaloss

    def __len__(self):
        return len(self.moire_list)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if torch.is_tensor(idx):
            idx = idx.tolist()

        moire_fn = self.moire_list[idx]
        clean_fn = self.clean_list[idx]

        moire_img = Image.open(moire_fn).convert('RGB')
        clean_img = Image.open(clean_fn).convert('RGB')

        if self.data_name == "uhdm":
            if self.is_training:
                if os.path.split(clean_fn)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                xxx = random.randint(0, w - 768)
                yyy = random.randint(0, h - 768)

                moire_img = moire_img.crop((xxx, yyy, xxx + 768, yyy + 768))
                clean_img = clean_img.crop((xxx, yyy, xxx + 768, yyy + 768))

            w, h = moire_img.size

        elif self.data_name == "fhdmi":
            if self.is_training:
                xxx = random.randint(0, 1408) # 1920 - 512
                yyy = random.randint(0, 568) # 1080 - 512

                moire_img = moire_img.crop((xxx, yyy, xxx + 512, yyy + 512))
                clean_img = clean_img.crop((xxx, yyy, xxx + 512, yyy + 512))

                # if random.random() > 0.5:
                #     moire_img = moire_img.transpose(Image.FLIP_LEFT_RIGHT)
                #     clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)

        elif self.data_name == "tip18":
            w, h = moire_img.size

            xxx = 0
            yyy = 0
            if self.is_training:
                xxx = random.randint(-6, 6)
                yyy = random.randint(-6, 6)

            moire_img = moire_img.crop((int(w / 6) + xxx, int(h / 6) + yyy, int(w * 5 / 6) + xxx, int(h * 5 / 6) + yyy))
            clean_img = clean_img.crop((int(w / 6) + xxx, int(h / 6) + yyy, int(w * 5 / 6) + xxx, int(h * 5 / 6) + yyy))

            moire_img = moire_img.resize((256, 256), Image.BILINEAR)
            clean_img = clean_img.resize((256, 256), Image.BILINEAR)

        elif self.data_name == "aim":
            if self.is_training:
                new_h, new_w = 512, 512

                xxx = random.randint(0, 511)
                yyy = random.randint(0, 511)

                moire_img = moire_img.crop((xxx, yyy, xxx + new_w, yyy + new_h))
                clean_img = clean_img.crop((xxx, yyy, xxx + new_w, yyy + new_h))

        # pixel rarity for the adaptive loss
        freq_img = np.array([-1])
        if self.adaloss:
            freq_img = clean_img.convert('L')
            freq_img = np.array(freq_img)
            freq_img = freq_img.reshape(-1)
            freq_img *= 255

        moire_img = transforms.ToTensor()(moire_img)
        clean_img = transforms.ToTensor()(clean_img)
        # freq_img = transforms.ToTensor()(freq_img)
        freq_img = torch.from_numpy(freq_img)

        sample = {"moire": moire_img, "clean": clean_img, "clean_freq": freq_img, "moire_fn": moire_fn, "clean_fn": clean_fn}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    main()
