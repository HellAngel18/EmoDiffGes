import pynvml
from utils import rotation_conversions as rc


import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from dataloaders import data_tools
from utils import config, logger_tools, other_tools, metric
import numpy as np


import warnings
warnings.filterwarnings('ignore')
from models.vq.model import RVQVAE

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
    
    def my_forward(self,motion_pred,motion_gt,mask) :
        loss = self.Loss(motion_pred[..., mask], motion_gt[..., mask])
        return loss



import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--body_part',type=str,default='whole')
    ## optimization
    parser.add_argument('--total-iter', default=80000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=400, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 200000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='outputs/rvqvae', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='RVQVAE', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=100, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    parser.add_argument('--mode', type=str, default='train', help='train or eval')
    
    
    return parser.parse_args()

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_{args.body_part}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



##### ---- Dataloader ---- #####
from dataloaders.mix_sep import CustomDataset
from utils.config import parse_args

dataset_args = parse_args("configs/beat2_rvqvae.yaml")
build_cache = True

trainSet = CustomDataset(dataset_args,"train",build_cache = build_cache)
testSet = CustomDataset(dataset_args,"test",build_cache = build_cache)
train_loader = torch.utils.data.DataLoader(trainSet,
                                              args.batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=8,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
test_loader = torch.utils.data.DataLoader(testSet,
                                          1,
                                            shuffle=False,
                                            num_workers=8,
                                            drop_last = True)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

train_loader_iter = cycle(train_loader)
test_loader_iter = cycle(test_loader)




joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
upper_body_mask = []
for i in joints:
    upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
mask = upper_body_mask
upper_rec_mask = list(range(len(mask)))

    


joints = list(range(25,55))
hands_body_mask = []
for i in joints:
    hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
mask = hands_body_mask
hand_rec_mask = list(range(len(mask)))






joints = [0,1,2,4,5,7,8,10,11]
lower_body_mask = []
for i in joints:
    lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
lower_body_mask.extend([330,331,332])
lower_mask = lower_body_mask
rec_mask = list(range(len(mask)))


joints = list(range(0,22))+list(range(25,55))
whole_body_mask = []
for i in joints:
    whole_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
whole_body_mask.extend([330,331,332])
mask = whole_body_mask
whole_rec_mask = list(range(len(mask)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_mask = mask = list(range(333, 433))
face_rec_mask = list(range(len(mask)))

eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
args.vae_layer = 4
args.vae_length = 240
args.vae_test_dim = 330
args.variational = False
args.data_path_1 = "./datasets/hub/"
args.vae_grow = [1,1,2,1]
eval_copy = getattr(eval_model_module, 'VAESKConv')(args).to(device)

other_tools.load_checkpoints(eval_copy, './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/'+'weights/AESKConv_240_100.bin', 'VAESKConv')
eval_copy.eval()
mean_pose = './mean_std/beatx_2_330_mean.npy'
std_pose = './mean_std/beatx_2_330_std.npy'

mean_pose = np.load(mean_pose)
std_pose = np.load(std_pose)

# load into torch cuda
mean_pose = torch.from_numpy(mean_pose).to(device)
std_pose = torch.from_numpy(std_pose).to(device)

##### ---- Network ---- #####
if args.body_part in "upper":
    dim_pose = 78   
elif args.body_part in "hands":
    dim_pose = 180
elif args.body_part in "lower":
    dim_pose = 54
elif args.body_part in "lower_trans":
    dim_pose = 57
elif args.body_part in "whole":
    dim_pose = 312
elif args.body_part in "whole_trans":
    dim_pose = 315
elif args.body_part in "face":
    dim_pose = 100

args.num_quantizers = 6
args.shared_codebook =  False
args.quantize_dropout_prob = 0.2

# upper
net1 = RVQVAE(args,
            78,
            args.nb_code,
            args.code_dim,
            args.code_dim,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)

# hands
net2 = RVQVAE(args,
            180,
            args.nb_code,
            args.code_dim,
            args.code_dim,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)

# lower_trans
net3 = RVQVAE(args,
            57,
            args.nb_code,
            args.code_dim,
            args.code_dim,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
            args.vq_act,
            args.vq_norm)



logger.info('loading checkpoint from {}'.format(args.resume_pth))
lower_ckpt = torch.load('ckpt/net_300000_lower.pth', map_location='cpu')
net3.load_state_dict(lower_ckpt['net'], strict=True)

upper_ckpt = torch.load('ckpt/net_300000_upper.pth', map_location='cpu')
net1.load_state_dict(upper_ckpt['net'], strict=True)

hands_ckpt = torch.load('ckpt/net_300000_hands.pth', map_location='cpu')
net2.load_state_dict(hands_ckpt['net'], strict=True)


if args.mode == 'test':
    net1.eval()
    net2.eval()
    net3.eval()
    net1.to(device)
    net2.to(device)
    net3.to(device)
    
    total_length = 0
    test_seq_list = testSet.selected_file
    align = 0 
    latent_out = []
    latent_ori = []
    diffs = []
    l2_all = 0 
    lvel = 0
    with torch.no_grad():
        for its, batch_data in enumerate(test_loader):
            gt_motion = batch_data.cuda().float()
            n = gt_motion.shape[1]
            remain = n%8
            
            if remain != 0:
                gt_motion = gt_motion[:, :-remain, :]

            gt_ori = gt_motion
            gt_upper_motion = gt_motion[...,upper_body_mask] # (bs, 64, dim)
            pred_upper_motion, loss_commit, perplexity = net1(gt_upper_motion).values()


            gt_hands_motion = gt_motion[...,hands_body_mask] # (bs, 64, dim)
            pred_hands_motion, loss_commit, perplexity = net2(gt_hands_motion).values()

            gt_lower_motion = gt_motion[...,lower_body_mask] # (bs, 64, dim)
            pred_lower_motion, loss_commit, perplexity = net3(gt_lower_motion).values()
            
            # replace the gt motion areas with the predicted motion
            pred_motion = gt_motion.clone()
            pred_motion[..., upper_body_mask] = pred_upper_motion
            pred_motion[..., hands_body_mask] = pred_hands_motion
            pred_motion[..., lower_body_mask] = pred_lower_motion

            diff = pred_motion - gt_motion
            rec_motion = pred_motion

            n = rec_motion.shape[1]

            rec_motion = rec_motion[..., :-103]
            gt_ori = gt_ori[..., :-103]
            
            rec_motion = rec_motion * std_pose + mean_pose
            gt_ori = gt_ori * std_pose + mean_pose
            
            remain = n%32
            latent_out.append(eval_copy.map2latent(rec_motion[:, :n-remain]).reshape(-1, 32).detach().cpu().numpy()) # bs * n/8 * 240
            latent_ori.append(eval_copy.map2latent(gt_ori[:, :n-remain]).reshape(-1, 32).detach().cpu().numpy())
            diffs.append(diff[:, :n-remain].reshape(-1, 32).detach().cpu().numpy())
            l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
            l2_all += l2_batch
        
        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        diffs_all = np.concatenate(diffs, axis=0)
        
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        print(f"fid: {fid}")

        print(f"L2 distance: {l2_all / len(test_loader)}")
