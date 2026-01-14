import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)


class GestureDiffusion(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Scheduler: {count_parameters(self.modality_encoder)}M')
    


    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        # i will do this
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        
        elif self.scheduler.config.prediction_type == "v_prediction":
            sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)
            alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
            pred_original_sample = alphas * sample - sigmas * model_output
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon



    def forward(self, cond_: dict) -> dict:

        audio = cond_['y']['audio']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        mask = cond_['y']['mask']
        style_feature = cond_['y']['style_feature']
        wavlm_feat = cond_['y']['wavlm']
        
        audio_feat = self.modality_encoder(audio, word, wavlm_feat)

        bs = audio_feat.shape[0]
        shape_ = (bs, 128 * 2, 1, 32)
        latents = torch.randn(shape_, device=audio_feat.device)

        latents = self._diffusion_reverse(latents, seed, audio_feat, guidance_scale=self.guidance_scale)

        return latents



    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            seed: torch.Tensor,
            at_feat: torch.Tensor,
            guidance_scale: float = 1,
    ) -> torch.Tensor:

        return_dict = {}
        # scale the initial noise by the standard deviation required by the scheduler, like in Stable Diffusion
        # this is the initial noise need to be returned for rectified training
        latents = latents * self.scheduler.init_noise_sigma

       
        noise = latents

        
        return_dict["init_noise"] = latents
        return_dict['at_feat'] = at_feat
        return_dict['seed'] = seed

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(at_feat.device)

        latents = torch.zeros_like(latents)
        
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in tqdm.tqdm(enumerate(timesteps)):
            latent_model_input = latents
            # actually it does nothing here according to ddim scheduler
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            model_output = self.denoiser.forward_with_cfg(
                x=latent_model_input,
                timesteps=t,
                seed=seed,
                at_feat=at_feat,
                guidance_scale=guidance_scale)

            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
        return_dict['latents'] = latents
        return return_dict
    
    def _diffusion_process(self, 
            latents: torch.Tensor, 
            audio_feat: torch.Tensor, 
            id: torch.Tensor, 
            seed: torch.Tensor, 
            mask: torch.Tensor, 
            style_feature: torch.Tensor
        ) -> dict:

        # [batch_size, n_frame, latent_dim]
        noise = torch.randn_like(latents)     # [bs, 384, 1, 32]
        bsz = latents.shape[0]

        
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )
        
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)

        model_output = self.denoiser(
            x=noisy_latents,
            timesteps=timesteps,
            seed=seed,
            at_feat=audio_feat
        )      #[bs, 384, 1, 32]

        latents_pred, noise_pred = self.predicted_origin(model_output, timesteps, noisy_latents)

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            "sample_pred": latents_pred,
            "sample_gt": latents,
        }
        return n_set
    
    def train_forward(self, cond_: dict, x0: torch.Tensor) -> dict:
        audio = cond_['y']['audio']        # [bs, 68266, 2]
        raw_audio = cond_['y']['wavlm']
        word = cond_['y']['word']       # [bs, T]
        id = cond_['y']['id']         # [bs, T, 1]
        seed = cond_['y']['seed']     # [bs, 4, 384]
        mask = cond_['y']['mask']     # [bs, 1, 1, 32]
        style_feature = cond_['y']['style_feature']
    
        audio_feat = self.modality_encoder(audio, word, raw_audio)       # [bs, 32, 768]
        n_set = self._diffusion_process(x0, audio_feat, id, seed, mask, style_feature)   #返回gt噪声和数据，以及预测的噪声和数据

        loss_dict = dict()

        # Diffusion loss
        if self.scheduler.config.prediction_type == "epsilon":
            model_pred, target = n_set['noise_pred'], n_set['noise']
        elif self.scheduler.config.prediction_type == "sample":
            model_pred, target = n_set['sample_pred'], n_set['sample_gt']
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")


        # mse loss
        diff_loss = self.masked_l2(target, model_pred, mask)

        loss_dict['diff_loss'] = diff_loss

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict


    def masked_l2(self, a, b, mask, reduction='mean'):
        loss = self.smooth_l1_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements

        if reduction == 'mean':
            mse_loss_val = mse_loss_val.mean()
        elif reduction == 'sum':
            mse_loss_val = mse_loss_val.sum()
        return mse_loss_val