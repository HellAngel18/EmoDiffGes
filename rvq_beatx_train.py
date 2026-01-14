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

dataset_args, _ = parse_args("configs/beat2_rvqvae.yaml")
build_cache = False

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



if args.body_part in "upper":
    joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
    upper_body_mask = []
    for i in joints:
        upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = upper_body_mask
    rec_mask = list(range(len(mask)))

    
elif args.body_part in "hands":

    joints = list(range(25,55))
    hands_body_mask = []
    for i in joints:
        hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = hands_body_mask
    rec_mask = list(range(len(mask)))


elif args.body_part in "lower":
    joints = [0,1,2,4,5,7,8,10,11]
    lower_body_mask = []
    for i in joints:
        lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    mask = lower_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part in "lower_trans":
    joints = [0,1,2,4,5,7,8,10,11]
    lower_body_mask = []
    for i in joints:
        lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    lower_body_mask.extend([330,331,332])
    mask = lower_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part in "whole_trans":
    joints = list(range(0,22))+list(range(25,55))
    whole_body_mask = []
    for i in joints:
        whole_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])
    whole_body_mask.extend([330,331,332])
    mask = whole_body_mask
    rec_mask = list(range(len(mask)))

elif args.body_part in "face":
    mask = list(range(333, 433))
    rec_mask = list(range(len(mask)))

eval_model_module = __import__(f"models.motion_representation", fromlist=["something"])
args.vae_layer = 4
args.vae_length = 240
args.vae_test_dim = 330
args.variational = False
args.data_path_1 = "./datasets/hub/"
args.vae_grow = [1,1,2,1]
eval_copy = getattr(eval_model_module, 'VAESKConv')(args).to('cuda')
other_tools.load_checkpoints(eval_copy, './datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/'+'weights/AESKConv_240_100.bin', 'VAESKConv')
eval_copy.eval()
mean_pose = './mean_std/beatx_2_330_mean.npy'
std_pose = './mean_std/beatx_2_330_std.npy'

mean_pose = np.load(mean_pose)
std_pose = np.load(std_pose)

# load into torch cuda
mean_pose = torch.from_numpy(mean_pose).cuda()
std_pose = torch.from_numpy(std_pose).cuda()

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
net = RVQVAE(args,
            dim_pose,
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


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

if args.mode == 'train':
    net.train()
    net.cuda()

    ##### ---- Optimizer & Scheduler ---- #####
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


    Loss = ReConsLoss(args.recons_loss)

    ##### ------ warm-up ------- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    for nb_iter in range(1, args.warm_up_iter):
        
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        
        gt_motion = next(train_loader_iter)


        gt_motion = gt_motion[...,mask].cuda().float() # (bs, 64, dim)

        pred_motion, loss_commit, perplexity = net(gt_motion).values()
        loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
        loss_vel = 0#Loss.my_forward(pred_motion, gt_motion,vel_mask)
        
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            
            logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    ##### ---- Training ---- #####
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    #best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)
    args.eval_iter = args.eval_iter * 10
    fid_his = []
    l2_his = []
    best_fid = 1000
    best_l2 = 0
    l2_history = []
    early_stop_patience = 5
    early_stop_counter = 0
    for nb_iter in range(1, args.total_iter + 1):
        
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion[...,mask].cuda().float() # bs, nb_joints, joints_dim, seq_len
        
        pred_motion, loss_commit, perplexity = net(gt_motion).values()
        loss_motion = Loss.my_forward(pred_motion, gt_motion,rec_mask)
        loss_vel = 0
        
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        
        if nb_iter % args.print_iter ==  0 :
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
            
            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

        # if nb_iter % args.eval_iter==0 :
        #     best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        # eval_trans.my_evaluation_vqvae(args.out_dir, val_loader, net, logger, writer)
        if nb_iter % args.eval_iter==0 :
            torch.save({'net' : net.state_dict()}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))

            net.eval()
            
            total_length = 0
            test_seq_list = testSet.selected_file
            align = 0 
            latent_out = []
            latent_ori = []
            diffs = []
            l2_all = 0 
            lvel = 0
            j = 55
            with torch.no_grad():
                for its, batch_data in enumerate(test_loader):
                    gt_motion = batch_data.cuda().float()
                    n = gt_motion.shape[1]
                    remain = n%8
                    
                    if remain != 0:
                        gt_motion = gt_motion[:, :-remain, :]

                    gt_ori = gt_motion
                    gt_motion = gt_motion[...,mask] # (bs, 64, dim)
                    bs = gt_motion.shape[0]
                    pred_motion, loss_commit, perplexity = net(gt_motion).values()
                    diff = pred_motion - gt_motion
                    
                    pred_motion = pred_motion
                    rec_motion = gt_ori
                    
                    rec_motion[..., mask] = pred_motion # it is already a 6d tensor
                    
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

                diffs_all = diffs_all.reshape(-1, len(mask))
                print(f"L2 distance: {l2_all / len(test_loader)}")
                

                # Add current L2 to history
                l2_history.append(l2_all)

                # Check for early stopping
                if len(l2_history) > early_stop_patience:
                    if best_l2 <= min(l2_history[-early_stop_patience-1:-1]):
                        early_stop_counter += 1
                        print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                        if early_stop_counter >= early_stop_patience:
                            print(f"Early stopping triggered at iteration {nb_iter}")
                            break
                    else:
                        early_stop_counter = 0


                if fid < best_fid:
                    best_fid = fid
                
                if l2_all > best_l2:
                    best_l2 = l2_all
                
                # do a ploting of the fid change given the history
                fid_his.append(fid)
                l2_his.append(l2_all)

                # use matplotlib to plot the fid_his and l2_his
                import matplotlib.pyplot as plt

                plt.figure()
                iterations = list(range(len(fid_his)))
                iterations = [x * args.eval_iter for x in iterations]
                plt.plot(iterations, fid_his, label='FID')
                plt.plot(iterations, l2_his, label='L2')
                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.title('FID and L2 over Iterations')
                plt.legend()
                plt.savefig(os.path.join(args.out_dir, 'fid_l2_plot.png'))
                
            
            net.train()

else:
    net.eval()
    net.cuda()
    
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
            gt_motion = gt_motion[...,mask] # (bs, 64, dim)
            pred_motion, loss_commit, perplexity = net(gt_motion).values()
            diff = pred_motion - gt_motion
            
            pred_motion = pred_motion
            rec_motion = gt_ori.clone()
            
            rec_motion[..., mask] = pred_motion # it is already a 6d tensor
            
            n = rec_motion.shape[1]

            rec_motion = rec_motion[..., :-103]
            gt_ori = gt_ori[..., :-103]
            
            rec_motion = rec_motion * std_pose + mean_pose
            gt_ori = gt_ori * std_pose + mean_pose
            
            remain = n%32
            latent_out.append(eval_copy.map2latent(rec_motion[:, :n-remain]).reshape(-1, 240).detach().cpu().numpy()) # bs * n/8 * 240
            latent_ori.append(eval_copy.map2latent(gt_ori[:, :n-remain]).reshape(-1, 240).detach().cpu().numpy())
            diffs.append(diff[:, :n-remain].reshape(-1, 32).detach().cpu().numpy())
            l2_batch = torch.sqrt(torch.sum(diff ** 2, dim=[1, 2])).mean().item()
            l2_all += l2_batch
        
        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        diffs_all = np.concatenate(diffs, axis=0)
        
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        print(f"fid: {fid}")

        print(f"L2 distance: {l2_all / len(test_loader)}")
