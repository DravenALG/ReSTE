import argparse
import os

parser = argparse.ArgumentParser(description='Binary Neural Network')

parser.add_argument(
    '--results_dir',
    default='./results',
    help='results dir  (default: ./results)')

parser.add_argument(
    '--save',
    default='',
    help='saved folder name in results dir (default: )')

# if we use resume, we will check whether save_path have the checkpoint and then load it.
# if we don't use resume, we should boost the basic config.
parser.add_argument(
    '--resume',
    action='store_true',
    help='whether resume to latest checkpoint')

# if we use evaluate, setting the model dir in --evaluate or -e, then the results will save in "./tmp".
parser.add_argument(
    '-e',
    '--evaluate',
    type=str,
    help='evaluate model on validation set')

parser.add_argument(
    '--seed',
    default=1234,
    type=int,
    help='the random seed (default: 1234)')

parser.add_argument(
    '-m',
    '--model',
    default='resnet20_1w1a',
    help='model architecture (default: resnet20_1w1a)')

parser.add_argument(
    '--dataset',
    default='cifar10',
    type=str,
    help='dataset - e.g cifar10, imagenet (default: cifar10)')

parser.add_argument(
    '--data_path',
    type=str,
    default='/data/cifar10',
    help='dictionary where the dataset is stored.  (default: /data/cifar10)')

parser.add_argument(
    '--ts_type',
    default='torch.cuda.FloatTensor',
    help='type of tensor')

# Training
parser.add_argument(
    '--gpus',
    default='0',
    help='gpus used for training - e.g 0,1,3 (default: 0)')

parser.add_argument(
    '--lr',
    default=0.1,
    type=float,
    help='learning rate (default: 0.1)')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='weight decay of loss. (default: 1e-4)')

parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='momentum value of sgd (default: 0.9)')

parser.add_argument(
    '--workers',
    default=8,
    type=int,
    help='number of data loading workers (default: 8)')

parser.add_argument(
    '--epochs',
    default=1000,
    type=int,
    help='number of total epochs to run (default: 1000)')

parser.add_argument(
    '--start_epoch',
    default=-1,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts) (default: -1) (we will begin in start_epoch+1)')

parser.add_argument(
    '-b',
    '--batch_size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size for training (default: 256)')

parser.add_argument(
    '-bt',
    '--batch_size_test',
    default=128,
    type=int,
    help='mini-batch size for testing (default: 128)')

parser.add_argument(
    '--print_freq',
    '-p',
    default=100,
    type=int,
    help='print frequency (default: 100)')

parser.add_argument(
    '--time_estimate',
    default=1,
    type=int,
    help='print estimating finish time,set to 0 to disable (default:1)')

parser.add_argument(
    '--a32',
    action='store_true',
    help='w1a32  (default: true)')

parser.add_argument(
    '--warm_up',
    action='store_true',
    help='whether use warm up')

parser.add_argument(
    '--estimator',
    default="ReSTE",
    type=str,
    help='what estimator we use')

parser.add_argument(
    '--o_end',
    default=3,
    type=float,
    help='hyper-parameters')

parser.add_argument(
    '--cal_ind',
    action='store_true',
    help='whether calculate the indicators of fitting error and stability')

args = parser.parse_args()