import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network

# basic
import socket
import warnings
import copy

# torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data as data
from tensorboardX import SummaryWriter

def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='augments', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--tmp_data_dir', type=str, default='augments', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--world_size', type=int, default=-1)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

args, unparsed = parser.parse_known_args()
jobid = os.environ["SLURM_JOBID"]
args.save = '{}/{}'.format(args.save, jobid)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    # For slurm available
    if "SLURM_NPROCS" in os.environ:
        # acquire world size from slurm
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = os.path.join(args.save, "dist_url." + jobid  + ".txt")
        if args.rank == 0:
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            with open(hostfile, "w") as f:
                f.write(args.dist_url)
        else:
            while not os.path.exists(hostfile):
                time.sleep(5)  # waite for the main process
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))

    # support multiple GPU on one node
    # assume each node have equal GPUs
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def worker(gpu, ngpus_per_node, config_in):
    # init
    args = copy.deepcopy(config_in)
    jobid = os.environ["SLURM_JOBID"]
    procid = int(os.environ["SLURM_PROCID"])
    args.gpu = gpu

    if args.gpu is not None:
        writer_name = "tb.{}-{:d}-{:d}".format(jobid, procid, gpu)
        logger_name = ".{}-{:d}-{:d}.aug.log".format(jobid, procid, gpu)
        ploter_name = "{}-{:d}-{:d}".format(jobid, procid, gpu)
        ck_name = "{}-{:d}-{:d}".format(jobid, procid, gpu)
    else:
        writer_name = "tb.{}-{:d}-all".format(jobid, procid)
        logger_name = "{}-{:d}-all.aug.log".format(jobid, procid)
        ploter_name = "{}-{:d}-all".format(jobid, procid)
        ck_name = "{}-{:d}-all".format(jobid, procid)

    writer = SummaryWriter(log_dir=os.path.join(args.save, writer_name))
    logger = utils.get_logger(os.path.join(args.save, logger_name))

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend="nccl", init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logger.info("args = %s", args)
    logger.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logger.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        # model = model.to(device)
        model.cuda(args.gpu)
        # When using a single GPU per process and per DistributedDataParallel, we need to divide
        # the batch size ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    best_acc_top1 = 0
    best_acc_top5 = 0

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc_top1 = checkpoint['best_acc_top1']
            logger.info("{} \n\n\n {}".format(checkpoint['state_dict'].keys(), model.state_dict().keys()))
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    data_dir = os.path.join(args.tmp_data_dir, 'imagenet')
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = data.distributed.DistributedSampler(train_data,
                                                        num_replicas=args.world_size,
                                                        rank=args.rank)
    valid_sampler = data.distributed.DistributedSampler(valid_data,
                                                        num_replicas=args.world_size,
                                                        rank=args.rank)
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True, num_workers=args.workers)

    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    lr = args.learning_rate
    for epoch in range(args.start_epoch, args.epochs):
        valid_sampler.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logger.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logger.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, epoch, writer, logger)
        logger.info('Train_acc: %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, logger)
        logger.info('Valid_acc_top1: %f', valid_acc_top1)
        logger.info('Valid_acc_top5: %f', valid_acc_top5)
        epoch_duration = time.time() - epoch_start
        logger.info('Epoch time: %ds.', epoch_duration)
        is_best = False
        if valid_acc_top5 > best_acc_top5:
            best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        if args.rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save)
    # get data with meta info

        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        


def train(train_queue, model, criterion, optimizer, epoch, writer, logger):
    global start_time
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()
    cur_step = epoch * len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/top1', prec1.item(), cur_step)
        writer.add_scalar('train/top5', prec5.item(), cur_step)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logger.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                                    step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logger.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg, duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main() 
