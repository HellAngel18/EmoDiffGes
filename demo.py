import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
from transformers import pipeline
from models.vq.model import RVQVAE

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

pipe = pipeline(
    "automatic-speech-recognition",
    model="/sata/public/lixr/EmoDiffGes/whisper-tiny.en",
    chunk_length_s=30,
    device=device,
)       

debug = False

class BaseTrainer(object):
    def __init__(self, args, cfg, ap):
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend
        tmp_dir = args.out_path + "custom/"+ time_name_expend + hf_dir
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/")
        self.audio_path = tmp_dir + "/tmp.wav"
        sf.write(self.audio_path, ap[1], ap[0])
        
        audio, ssr = librosa.load(self.audio_path, sr=args.audio_sr)

        # 处理tmp.lab文件
        file_path = tmp_dir + "/tmp.lab"
        if not os.path.exists(file_path):
            logger.info("No tmp.lab found, generating with ASR...")
            if not debug:
                text = pipe(audio, batch_size=8)["text"]
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(text)
        else:
            logger.info(f"Using existing tmp.lab at {file_path}")
        
        # 设置TextGrid文件路径
        self.textgrid_path = args.textgrid_file_path
        if self.textgrid_path and os.path.exists(self.textgrid_path):
            logger.info(f"Using TextGrid file at {self.textgrid_path}")
        else:
            logger.warning(f"No valid TextGrid file provided at {self.textgrid_path}. Skipping TextGrid processing.")
            self.textgrid_path = None

        ap = (ssr, audio)
        self.args = args
        self.rank = 0  # dist.get_rank()
       
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
    
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {cfg.model.g_name} success")

        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()    

        self.args = args
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])

        ##### VQ-VAE models #####
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        self.vq_model_face = self._create_face_vq_model(vq_model_module)
        
        self.vq_models = self._create_body_vq_models()
        
        self.vq_model_face.eval().to(self.rank)
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 

        self.args.vae_length = 240
        
        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        
        ##### Normalization #####
        self.use_trans = self.args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).cuda())
        
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()
    
    def _create_face_vq_model(self, module):
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        model = getattr(module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(model, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", 
                                   self.args.e_name)
        return model
    
    def _create_body_vq_models(self):
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config['dim_pose'], part)
            vq_models[part] = model
            
        return vq_models
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        return model
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg

        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        style_feature = None
        
        return {
            "in_audio": in_audio,
            "wavlm": wavlm,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "latent_in": latent_in,
            "tar_id": tar_id,
            "tar_contact": tar_contact,
            "style_feature": style_feature,
        }
    
    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         
        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] = in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

            shape_ = (bs, 3*128, 1, 32)
            sample = self.model(cond_)['latents']
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            last_sample = sample.clone()
            
            rec_latent_upper = sample[...,:128]
            rec_latent_hands = sample[...,128:2*128]
            rec_latent_lower = sample[...,2*128:]
            
            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1] = rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower

        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_exps = tar_exps
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape (bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }

    def test_demo(self, epoch, audio_name):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        self.model.eval()
        self.smplx.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                # 使用音频文件名作为输出文件名，避免覆盖
                results_npz_file_save_path = results_save_path + f"result_{audio_name}_{self.time_name_expend}.npz"
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_npz_file_save_path, 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.args,
                )

        result = [render_vid_path, results_npz_file_save_path]
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result
       
@logger.catch
def emodiffges(audio_path, sample_stratege, textgrid_path=None):
    args, cfg = config.parse_args()
    
    # 设置TextGrid文件路径
    args.textgrid_file_path = textgrid_path
    if textgrid_path and not os.path.exists(textgrid_path):
        logger.warning(f"TextGrid file {textgrid_path} does not exist. Proceeding without TextGrid.")
        args.textgrid_file_path = None
    
    print(sample_stratege)

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # 加载音频数据
    audio_data, sr = librosa.load(audio_path, sr=args.audio_sr)
    ap = (sr, audio_data)

    trainer = BaseTrainer(args, cfg, ap=ap)
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    # 使用音频文件名作为标识
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    result = trainer.test_demo(999, audio_name)
    return result

def main():
    base_path = "/sata/public/lixr/EmoDiffGes/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/wave16k/"
    # 定义音频和TextGrid文件对
    audio_textgrid_pairs = [
        (os.path.join(base_path, "2_scott_0_73_73.wav"), "/sata/public/lixr/EmoDiffGes/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/textgrid/2_scott_0_73_73.TextGrid")  # 替换为实际TextGrid路径
    ]

    sample_stratege = "DDPM"  # 默认采样策略
    for audio_path, textgrid_path in audio_textgrid_pairs:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file {audio_path} does not exist. Skipping.")
            continue
        logger.info(f"Processing audio: {audio_path}, TextGrid: {textgrid_path}")
        result = emodiffges(audio_path, sample_stratege, textgrid_path)
        logger.info(f"Results saved: Video={result[0]}, NPZ={result[1]}")

if __name__ == "__main__":
    main()

'''import os
import signal
import time
import csv
import sys
import warnings
import random
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer, other_tools
from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
import subprocess
from transformers import pipeline
from models.vq.model import RVQVAE

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

pipe = pipeline(
  "automatic-speech-recognition",
  model="/sata/public/lixr/EmoDiffGes/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)       

debug = False

class BaseTrainer(object):
    def __init__(self, args, cfg, ap):
        
        hf_dir = "hf"
        time_local = time.localtime()
        time_name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        self.time_name_expend = time_name_expend
        tmp_dir = args.out_path + "custom/"+ time_name_expend + hf_dir
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/")
        self.audio_path = tmp_dir + "/tmp.wav"
        sf.write(self.audio_path, ap[1], ap[0])
        
        
        audio, ssr = librosa.load(self.audio_path,sr=args.audio_sr)

        # 使用现有的TextGrid文件
        file_path = tmp_dir + "/tmp.lab"
        self.textgrid_path = "/path/to/your/tmp.TextGrid"  # 替换为你的TextGrid文件路径

        if not os.path.exists(file_path):
            logger.info("No tmp.lab found, generating with ASR...")
            if not debug:
                text = pipe(audio, batch_size=8)["text"]
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(text)
        else:
            logger.info(f"Using existing tmp.lab at {file_path}")
        

        

        ap = (ssr, audio)
        self.args = args
        self.rank = 0 # dist.get_rank()
       
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
    
    
        self.rank = 0 # dist.get_rank()
       
        self.checkpoint_path = tmp_dir
        args.tmp_dir = tmp_dir
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{cfg.model.model_name}", fromlist=["something"])
        
        self.model = torch.nn.DataParallel(getattr(model_module, cfg.model.g_name)(cfg), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {cfg.model.g_name} success")

        self.smplx = smplx.create(
        self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()    

        self.args = args
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])

        
        ##### VQ-VAE models #####
        """Initialize and load VQ-VAE models for different body parts."""
        # Face VQ model
        vq_model_module = __import__("models.motion_representation", fromlist=["something"])
        self.vq_model_face = self._create_face_vq_model(vq_model_module)
        
        # Body part VQ models
        self.vq_models = self._create_body_vq_models()
        
        # Set all VQ models to eval mode
        self.vq_model_face.eval().to(self.rank)
        for model in self.vq_models.values():
            model.eval().to(self.rank)
        self.vq_model_upper, self.vq_model_hands, self.vq_model_lower = self.vq_models.values()
        self.vqvae_latent_scale = self.args.vqvae_latent_scale 


        self.args.vae_length = 240
        
        ##### Loss functions #####
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        
        
        ##### Normalization #####
        self.use_trans = self.args.use_trans
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        # Extract body part specific normalizations
        for part in ['upper', 'hands', 'lower']:
            mask = globals()[f'{part}_body_mask']
            setattr(self, f'mean_{part}', torch.from_numpy(self.mean[mask]).cuda())
            setattr(self, f'std_{part}', torch.from_numpy(self.std[mask]).cuda())
        
        # Translation normalization if needed
        if self.args.use_trans:
            self.trans_mean = torch.from_numpy(np.load(self.args.mean_trans_path)).cuda()
            self.trans_std = torch.from_numpy(np.load(self.args.std_trans_path)).cuda()
    
    def _create_face_vq_model(self, module):
        """Create and initialize face VQ model."""
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        model = getattr(module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(model, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", 
                                   self.args.e_name)
        return model
    
    def _create_body_vq_models(self):
        """Create VQ-VAE models for body parts."""
        vq_configs = {
            'upper': {'dim_pose': 78},
            'hands': {'dim_pose': 180},
            'lower': {'dim_pose': 54 if not self.args.use_trans else 57}
        }

        vq_models = {}
        for part, config in vq_configs.items():
            model = self._create_rvqvae_model(config['dim_pose'], part)
            vq_models[part] = model
            
        return vq_models
    
    def _create_rvqvae_model(self, dim_pose: int, body_part: str) -> RVQVAE:
        """Create a single RVQVAE model with specified configuration."""
        args = self.args
        model = RVQVAE(
            args, dim_pose, args.nb_code, args.code_dim, args.code_dim,
            args.down_t, args.stride_t, args.width, args.depth,
            args.dilation_growth_rate, args.vq_act, args.vq_norm
        )
        
        # Load pretrained weights
        checkpoint_path = getattr(args, f'vqvae_{body_part}_path')
        model.load_state_dict(torch.load(checkpoint_path)['net'])
        return model
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank)
        if 'wavlm' in dict_data:
            wavlm = dict_data["wavlm"].to(self.rank)
        else:
            wavlm = None
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg

        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
      

        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        style_feature = None
        
        return {
            "in_audio": in_audio,
            "wavlm": wavlm,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "tar_contact": tar_contact,
            "style_feature":style_feature,
        }
    
    def _g_test(self, loaded_data):
        
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]

            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

            shape_ = (bs, 3*128, 1, 32)
            sample = self.model(cond_)['latents']
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            last_sample = sample.clone()
            
            rec_latent_upper = sample[...,:128]
            rec_latent_hands = sample[...,128:2*128]
            rec_latent_lower = sample[...,2*128:]
            
           

            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1]=rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower




        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]


        rec_exps = tar_exps
        #rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }


    def test_demo(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                

                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load("./demo/examples/2_scott_0_1_1.npz", allow_pickle=True)

                results_npz_file_save_path = results_save_path+f"result_{self.time_name_expend}"+'.npz'
                np.savez(results_npz_file_save_path,
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_npz_file_save_path, 
                    # results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.args,
                    )

        result = [
            gr.Video(value=render_vid_path, visible=True),
            gr.File(value=results_npz_file_save_path, label="download motion and visualize in blender"),
            ]
        
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result
       
@logger.catch
def emodiffges(audio_path,sample_stratege):
    args, cfg = config.parse_args()
    
    print(sample_stratege)

    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    #logger_tools.set_args_and_logger(args, rank)
    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # return one intance of trainer
    trainer = BaseTrainer(args, cfg, ap = audio_path)
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    
    result = trainer.test_demo(999)
    return result

base_path = "/sata/public/lixr/EmoDiffGes/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/wave16k/"
examples = [
    [os.path.join(base_path, "2_scott_0_73_73.wav"), "DDPM", "/sata/public/lixr/EmoDiffGes/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/textgrid/2_scott_0_73_73.TextGrid"],
    [os.path.join(base_path, "2_scott_0_2_2.wav"), "DDPM", None],
    [os.path.join(base_path, "2_scott_0_3_3.wav"), "DDPM", None],
    [os.path.join(base_path, "2_scott_0_4_4.wav"), "DDPM", None],
    [os.path.join(base_path, "2_scott_0_5_5.wav"), "DDPM", None],
]

demo = gr.Interface(
    emodiffges,  # function
    inputs=[
        gr.Audio(),
        gr.Radio(
            choices=["DDPM", "其他"],  # 根据实际支持的采样策略调整
            label="采样策略",
            value="DDPM",  # 默认值，与描述中推荐的策略一致
        ),
    ],  # input type
    outputs=[
        gr.Video(format="mp4", visible=True),
        gr.File(label="download motion and visualize in blender")
    ],
    title='Emotion-Aware Co-Speech Holistic Gesture Generation with Progressive Synergistic Diffusion',
    description="1. Upload your audio.  <br/>\
        2. Then, sit back and wait for the rendering to happen! This may take a while (e.g. 1-4 minutes) <br/>\
        3. After, you can view the videos.  <br/>\
        4. Notice that we use a fix face animation, our method only produce body motion. <br/>\
        5. Use DDPM sample strategy will generate a better result, while it will take more inference time.  \
            ",
    article="Project links: [EmoDiffGes](https://github.com/HellAngel18/EmoDiffGes). <br/>\
             Reference links: [EMAGE](https://pantomatrix.github.io/EMAGE/). ", 
    examples=examples,
)

            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.3'
    os.environ["MASTER_PORT"]='8678'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    demo.launch(server_name="0.0.0.0",share=True)'''