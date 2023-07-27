# python build-in library and args
import math
import os
import argparse
import time
import logging
import random
import sys
import datetime

# arguments
from utils.arguments import args
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# pytorch library
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

# project files
from dataset import load_data
from model import models_cifar
from model import models_imagenet
from model import binary_module
from utils.tools import *


def main():
    global args, best_prec1, conv_modules

    best_prec1 = 0

    # setting base seed
    random.seed(args.seed)

    # setting save path
    if args.evaluate:
        args.results_dir = './tmp'
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.resume:
        # saving the arguments settings
        with open(os.path.join(save_path, 'argument.txt'), 'w') as args_file:
            args_file.write(str(datetime.datetime.now()) + '\n\n')
            for args_n, args_v in args.__dict__.items():
                args_v = '' if not args_v and not isinstance(args_v, int) else args_v
                args_file.write(str(args_n) + ':  ' + str(args_v) + '\n')

        # output log to log file
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=os.path.join(save_path, 'logger.log'),
                            filemode='w')

        # output log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        logging.info("saving to %s", save_path)
        logging.debug("run arguments: %s", args)
    else:
        # output log to log file
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=os.path.join(save_path, 'logger.log'),
                            filemode='a')

        # output log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    # gpu setting
    args.gpus = [int(i) for i in args.gpus.split(',')]
    cudnn.benchmark = True

    # dataset setting
    if args.dataset == 'imagenet':
        num_classes = 1000
        model_zoo = 'models_imagenet.'
    elif args.dataset == 'cifar10':
        num_classes = 10
        model_zoo = 'models_cifar.'

    # loading model
    if len(args.gpus) == 1:
        model = eval(model_zoo + args.model)(num_classes=num_classes).cuda()
    else:
        model = nn.DataParallel(eval(model_zoo + args.model)(num_classes=num_classes).cuda())
    if not args.resume:
        # output the model information
        logging.info("creating model %s", args.model)
        logging.info("model structure: %s", model)
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

    # load checkpoint information
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            logging.error('invalid checkpoint: {}'.format(args.evaluate))
        else:
            checkpoint = torch.load(args.evaluate)
            if len(args.gpus) > 1:
                checkpoint['state_dict'] = add_module_fromdict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)", args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = os.path.join(save_path, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            if len(args.gpus) > 1:
                checkpoint['state_dict'] = add_module_fromdict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)", checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    # load loss
    criterion = nn.CrossEntropyLoss().cuda()

    # evaluate
    if args.evaluate:
        # load data and test
        val_loader = load_data.load(
            type='val',
            dataset=args.dataset,
            data_path=args.data_path,
            batch_size=args.batch_size,
            batch_size_test=args.batch_size_test,
            num_workers=args.workers)
        with torch.no_grad():
            val_loss, val_prec1, val_prec5, _, _ = validate(val_loader, model, criterion, 0)
        logging.info('\n Validation loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(val_loss=val_loss, val_prec1=val_prec1, val_prec5=val_prec5))
        return

    # load_data
    train_loader, val_loader = load_data.load(
        type='both',
        dataset=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        num_workers=args.workers)

    # load optimizer
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)


    # if resume, compute the learning rate of current epoch
    def cosin(i, T, emin=0, emax=0.01):
        return emin + (emax - emin) / 2 * (1 + np.cos(i * np.pi / T))

    if args.resume:
        if args.warm_up:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cosin(args.start_epoch - args.warm_up * 4, args.epochs - args.warm_up * 4, 0,
                                          args.lr)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cosin(args.start_epoch, args.epochs, 0, args.lr)

    # setup learning rate decay strategy
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm_up * 4, eta_min=0,
                                                                  last_epoch=args.start_epoch - args.warm_up * 4)

    if not args.resume:
        # output the optimization information
        logging.info("criterion: %s", criterion)
        logging.info('scheduler: %s', lr_scheduler)


    def ReSTE(epoch):
        ratio = torch.tensor(epoch / args.epochs)

        # compute o
        beta = 1 - torch.cos(ratio * math.pi * 0.5)  # cos
        o = torch.tensor(1 + beta * (args.o_end - 1))
        o_a = torch.tensor(1 + beta * (args.o_end - 1))
        return o, o_a

    # setup conv_modules.epoch
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_modules.append(module)

    # training
    if args.cal_ind:
        total_estimating_error = 0
        total_stability = 0

    for epoch in range(args.start_epoch + 1, args.epochs):
        time_start = datetime.datetime.now()
        # warm up
        if args.warm_up and epoch < 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / 5
        for param_group in optimizer.param_groups:
            logging.info('lr: %s', param_group['lr'])

        # compute o and t in back-propagation and add to modules
        o, o_a = ReSTE(epoch)
        logging.info(f"o is {o},  o_a is {o_a}")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.o = o.cuda()
                module.o_a = o_a.cuda()

        for module in conv_modules:
            module.epoch = epoch

        # train one epoch
        train_loss, train_prec1, train_prec5, estimating_error, stability_var = train(
            train_loader, model, criterion, epoch, optimizer)

        # adjust Lr
        if epoch >= 4 * args.warm_up:
            lr_scheduler.step()

        # evaluate
        with torch.no_grad():
            for module in conv_modules:
                module.epoch = -1
            val_loss, val_prec1, val_prec5, _, _ = validate(
                val_loader, model, criterion, epoch)

        # remember best prec
        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = max(val_prec1, best_prec1)
            best_epoch = epoch
            best_loss = val_loss

        # save model with best accuracy
        if epoch % 1 == 0:
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            model_parameters = model.module.parameters() if len(args.gpus) > 1 else model.parameters()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model_state_dict,
                'best_prec1': best_prec1,
                'parameters': list(model_parameters),
            }, is_best, path=save_path)

        if args.time_estimate > 0 and epoch % args.time_estimate == 0:
            # measure the cost time in current epoch and estimate the final finish time
            time_end = datetime.datetime.now()
            cost_time, finish_time = get_time(time_end - time_start, epoch, args.epochs)
            logging.info('Time cost: ' + cost_time + '\t' 'Time of Finish: ' + finish_time)

        # logging the results
        logging.info('\n Epoch: {0}\n'
                     'Training loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \n'
                     'Validation loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        if args.cal_ind:
            total_estimating_error = total_estimating_error + estimating_error
            total_stability = total_stability + stability_var

            # logging the indicators of fitting error and gradient stability
            logging.info('Estimating_Error {estimating_error:.3f} \t'
                     'Gradient Stability {stability:.8f} \n'
                         .format(estimating_error=estimating_error, stability=stability_var))

    logging.info('*' * 50 + 'DONE' + '*' * 50)
    logging.info('\n Best_Epoch: {0}\t'
                 'Best_Prec1 {prec1:.4f} \t'
                 'Best_Loss {loss:.3f} \t'
                 .format(best_epoch + 1, prec1=best_prec1, loss=best_loss))

    if args.cal_ind:
        logging.info('\n Total Estimating_Error: {total_estimating_error:.4f} \t'
                     'Total stability_var: {total_stability} \n'
                     .format(total_estimating_error=total_estimating_error / args.epochs,
                             total_stability=total_stability))


# this function imitate a forward pass of training or testing, using training parameter
def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    # init all the AverageMeter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time of current batch
        data_time.update(time.time() - end)
        if i == 1 and training:
            for module in conv_modules:
                module.epoch = -1
        input_var = inputs.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss_final = loss

        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

        if training and args.cal_ind:
            if "vgg" in args.model:
                estimating_error = get_fitting_error_vgg(model, args.o_end)
            else:
                estimating_error = get_fitting_error(model, args.o_end)

            if "vgg" in args.model:
                stability_var = get_stability_var_vgg(model)
            else:
                stability_var = get_stability_var(model)
        else:
            estimating_error = None
            stability_var = None

        # measure train time of current batch
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time,
                data_time=data_time, loss=losses,
                top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg, estimating_error, stability_var


def train(data_loader, model, criterion, epoch, optimizer):
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion,  epoch):
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == "__main__":
    main()
