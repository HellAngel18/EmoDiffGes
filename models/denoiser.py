import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers.utils import *
from .layers.transformer import SpatialTemporalBlock, CrossAttentionBlock

class GestureDenoiser(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        n_seed=8,
        flip_sin_to_cos= True,
        freq_shift = 0,
        cond_proj_dim=None,
        use_exp=True,
    
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_exp = use_exp
        self.joint_num = 3 if not self.use_exp else 4
        self.body_joint_num = 3
        self.upper_joint_num = 1
        self.hands_joint_num = 1
        self.lower_joint_num = 1
        self.face_joint_num = 1
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.body_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.body_joint_num,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.lower_1_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.lower_joint_num,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.hands_1_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.hands_joint_num,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])

        self.upper_1_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.upper_joint_num,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        
        self.face_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.face_joint_num,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.upper_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.upper_joint_num,context_dim=latent_dim*self.joint_num*2,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.hands_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.hands_joint_num,context_dim=latent_dim*self.joint_num*2,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.lower_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.lower_joint_num,context_dim=latent_dim*self.joint_num*2,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.whole_cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim,context_dim=latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.mytimmblocks = nn.ModuleList([
            SpatialTemporalBlock(dim=self.latent_dim,num_heads=self.num_heads,enable_spatial=False,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)])
        
        self.face_mytimmblocks = nn.ModuleList([
            SpatialTemporalBlock(dim=self.latent_dim,num_heads=self.num_heads,enable_spatial=False,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)])
            
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        
        #self.embed_text = nn.Linear(self.input_dim*(self.body_joint_num+self.face_joint_num)*4, self.latent_dim)
        self.embed_text = nn.Linear(self.input_dim*2*4, self.latent_dim)

        self.body_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.face_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.whole_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.upper_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.hands_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.lower_output_process = OutputProcess(self.input_dim, self.latent_dim)
        self.hands_lower_output_process = OutputProcess(self.input_dim, self.latent_dim)

        self.rel_pos = SinusoidalEmbeddings(self.latent_dim)
        self.face_input_process = InputProcess(self.input_dim, self.latent_dim)
        self.upper_input_process = InputProcess(self.input_dim, self.latent_dim)
        self.hands_input_process = InputProcess(self.input_dim, self.latent_dim)
        self.lower_input_process = InputProcess(self.input_dim, self.latent_dim)
        self.whole_input_process = InputProcess(self.input_dim, self.latent_dim)
        self.input_process = InputProcess(self.input_dim , self.latent_dim)
        self.face_input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.upper_input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.hands_input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.lower_input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.whole_input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)

        
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim, self.activation, cond_proj_dim=cond_proj_dim, zero_init_cond=True)
        time_dim = self.latent_dim
        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        if cond_proj_dim is not None:
            self.cond_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        
        self.null_cond_embed = nn.Parameter(torch.zeros(32, self.latent_dim*self.joint_num, requires_grad=True))

        self.face_cond_proj = nn.Linear(self.latent_dim, self.latent_dim * self.joint_num)
        self.upper_cond_proj = nn.Linear(self.latent_dim, self.latent_dim * self.joint_num)
        self.hands_cond_proj = nn.Linear(self.latent_dim, self.latent_dim * self.joint_num)
        self.lower_cond_proj = nn.Linear(self.latent_dim, self.latent_dim * self.joint_num)

    # dropout mask
    def prob_mask_like(self, shape, prob, device):
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
    


    @torch.no_grad()
    def forward_with_cfg(self, x, timesteps, seed, at_feat, cond_time=None, guidance_scale=1):
        """
        Forward pass with classifier-free guidance.
        Args:
            x: [batch_size, njoints, nfeats, max_frames]
            timesteps: [batch_size]
            seed: the previous gesture segment
            at_feat: the audio feature
            guidance_scale: Scale for classifier-free guidance (1.0 means no guidance)
        """
        # Run both conditional and unconditional in a single forward pass
        if guidance_scale > 1:
            output = self.forward(
                x,
                timesteps,
                seed,
                at_feat,
                cond_time=cond_time,
                cond_drop_prob=0.0,
                null_cond=False,
                do_classifier_free_guidance=True
            )
            # Split predictions and apply guidance
            pred_cond, pred_uncond = output.chunk(2, dim=0)
            guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
        else:
            guided_output = self.forward(x, timesteps, seed, at_feat, cond_time=cond_time, cond_drop_prob=0.0, null_cond=False)
        
        return guided_output
    


    def forward(self, x, timesteps, seed, at_feat, cond_time=None, cond_drop_prob: float = 0.1, null_cond=False, do_classifier_free_guidance=False, force_cfg=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        do_classifier_free_guidance: whether to perform classifier-free guidance (doubles batch)
        """
        _,_,_,noise_length = x.shape

        #x_body = x[:, :384, :, :]                       # [bs,384,1,32]
        #x_upper = x[:, :128, :, :]                      # [bs,128,1,32]
        #x_hands = x[:, 128:256, :, :]                   # [bs,128,1,32]
        #x_lower = x[:, 256:384, :, :]                   # [bs,128,1,32]
        x_whole = x[:, :128, :, :]
        x_face = x[:, 128:, :, :]                       # [bs,128,1,32]

        #if x_body.shape[2] == 1:
        #    x_body = x_body.squeeze(2)    # [bs, 384, 32]   
        #    x_face = x_face.squeeze(2)    # [bs, 128, 32]
        #    x_body = x_body.reshape(x_body.shape[0], self.body_joint_num, -1, x_body.shape[2])    # [bs, 3, 128, 32] 
        #    x_face = x_face.reshape(x_face.shape[0], self.face_joint_num, -1, x_face.shape[2])    # [bs, 1, 128, 32]

        #x_upper = x_upper.squeeze(2)       # [bs,128,32]
        #x_hands = x_hands.squeeze(2)       # [bs,128,32]
        #x_lower = x_lower.squeeze(2)       # [bs,128,32]
        x_face = x_face.squeeze(2)         # [bs,128,32]
        #x_upper = x_upper.reshape(x_upper.shape[0], self.upper_joint_num, -1, x_upper.shape[2])    # [bs,1,128,32]
        #x_hands = x_hands.reshape(x_hands.shape[0], self.hands_joint_num, -1, x_hands.shape[2])    # [bs,1,128,32]
        #x_lower = x_lower.reshape(x_lower.shape[0], self.lower_joint_num, -1, x_lower.shape[2])    # [bs,1,128,32]
        x_whole = x_whole.squeeze(2)
        x_whole = x_whole.reshape(x_whole.shape[0], 1, -1, x_whole.shape[2])
        x_face = x_face.reshape(x_face.shape[0], self.face_joint_num, -1, x_face.shape[2])         # [bs,1,128,32]
            
        # Double the batch for classifier free guidance
        if do_classifier_free_guidance and not self.training:
            #x_body = torch.cat([x_body] * 2, dim=0)
            #x_upper = torch.cat([x_upper] * 2, dim=0)
            #x_hands = torch.cat([x_hands] * 2, dim=0)
            #x_lower = torch.cat([x_lower] * 2, dim=0)
            x_whole = torch.cat([x_whole] * 2, dim=0)
            x_face = torch.cat([x_face] * 2, dim=0)
            seed = torch.cat([seed] * 2, dim=0)
            at_feat = torch.cat([at_feat] * 2, dim=0)
       
        #bs, body_joints, body_nfeats, nframes = x_body.shape      # [bs, 3, 128, 32]
        #bs, upper_joints, upper_nfeats, nframes = x_upper.shape    
        #_, hands_joints, hands_nfeats, _ = x_hands.shape
        #_, lower_joints, lower_nfeats, _ = x_lower.shape
        bs, whole_joints, whole_nfeats, nframes = x_whole.shape
        _, face_joints, face_nfeats, _ = x_face.shape             # [bs, 1, 128, 32]
        
        # need to be an arrary, especially when bs is 1
        timesteps = timesteps.expand(bs).clone()         # [bs]
        time_emb = self.time_proj(timesteps)     # [bs, 256]
        time_emb = time_emb.to(dtype=x.dtype)

        if cond_time is not None and self.cond_proj is not None:
            cond_time = cond_time.expand(bs).clone()
            cond_emb = self.cond_proj(cond_time)
            cond_emb = cond_emb.to(dtype=x.dtype)
            emb_t = self.time_embedding(time_emb, cond_emb)
        else:
            emb_t = self.time_embedding(time_emb)    # [bs, 256]
        
        if self.n_seed != 0:
            embed_text = self.embed_text(seed.reshape(bs, -1))     # [128, 4, 512] -> [128, 256]
            emb_seed = embed_text
        
        # Handle both conditional and unconditional branches in a single forward pass
        if do_classifier_free_guidance and not self.training:
            # First half of batch: conditional, Second half: unconditional
            null_cond_embed = self.null_cond_embed.to(at_feat.dtype)
            at_feat_uncond = null_cond_embed.unsqueeze(0).expand(bs//2, -1, -1)
            at_feat = torch.cat([at_feat[:bs//2], at_feat_uncond], dim=0)
        else:
            if force_cfg is None:
                if self.training:
                    keep_mask = self.prob_mask_like((bs,), 1 - cond_drop_prob, device=at_feat.device)      # [bs]
                    keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")    #[bs, 1, 1]
                    
                    null_cond_embed = self.null_cond_embed.to(at_feat.dtype)    # [32, 1024]
                    at_feat = torch.where(keep_mask_embed, at_feat, null_cond_embed)    # [bs, 32, 1024]     #音频文本进行随机掩码

                if null_cond:
                    at_feat = self.null_cond_embed.to(at_feat.dtype).unsqueeze(0).expand(bs, -1, -1)
            else:
                force_cfg = torch.tensor(force_cfg, device=at_feat.device)
                force_cfg_embed = rearrange(force_cfg, "b -> b 1 1")

                null_cond_embed = self.null_cond_embed.to(at_feat.dtype)
                at_feat = torch.where(force_cfg_embed, at_feat, null_cond_embed)

        
        x_face_seq = self.face_input_process(x_face)        # [bs, 1, 32, 256]
        face_cond = self.face_cond_proj(x_face_seq.mean(dim=1))   # [bs, 32, 1024]


        #x_body_seq = self.input_process(x_body)   # [bs, 3, 128, 32] -> [bs, 3, 32, 256]
        #x_upper_seq = self.upper_input_process(x_upper)
        #x_hands_seq = self.hands_input_process(x_hands)
        #x_lower_seq = self.lower_input_process(x_lower)
        x_whole_seq = self.whole_input_process(x_whole)



        # add the seed information
        #embed_style_body = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.body_joint_num, 32, -1)    # [bs, 3, 32, 256] (300, 256)
        #x_body_seq = torch.cat([embed_style_body, x_body_seq], axis=-1)  # -> [bs, 3, 32, 512]
        #x_body_seq = self.input_process2(x_body_seq)     # [bs, 3, 32, 256]

        embed_style = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2)   # [bs,1,1,256]
        embed_style_face = embed_style.expand(-1, self.face_joint_num, 32, -1)       #[bs,1,32,256]
        #embed_style_upper = embed_style.expand(-1, self.upper_joint_num, 32, -1)      #[bs,1,32,256]
        #embed_style_hands = embed_style.expand(-1, self.hands_joint_num, 32, -1)      # [bs,1,32,256]
        #embed_style_lower = embed_style.expand(-1, self.lower_joint_num, 32, -1)      #[bs,1,32,256]
        embed_style_whole = embed_style.expand(-1, 1, 32, -1)

        #embed_style_face = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.face_joint_num, 32, -1)    # [bs, 1, 32, 256] (300, 256)
        x_face_seq = torch.cat([embed_style_face, x_face_seq], axis=-1)     # [bs, 1, 32, 512] 
        x_face_seq = self.face_input_process2(x_face_seq)       ## [bs, 1, 32, 256] 

        #x_upper_seq = torch.cat([embed_style_upper, x_upper_seq], axis=-1)     #[bs,1,32,512]
        #x_upper_seq = self.upper_input_process2(x_upper_seq)      #[bs,1,32,256]

        #x_hands_seq = torch.cat([embed_style_hands, x_hands_seq], axis=-1)      #[bs,1,32,512]
        #x_hands_seq = self.hands_input_process2(x_hands_seq)

        #x_lower_seq = torch.cat([embed_style_lower, x_lower_seq], axis=-1)
        #x_lower_seq = self.lower_input_process2(x_lower_seq)

        x_whole_seq = torch.cat([embed_style_whole, x_whole_seq], axis=-1)
        x_whole_seq = self.whole_input_process2(x_whole_seq)

        

        # apply the positional encoding
        #x_body_seq = x_body_seq.reshape(bs * self.body_joint_num, nframes, -1)     # [384, 32, 256]
        #pos_emb = self.rel_pos(x_body_seq)        #[32, 256]
        #x_body_seq, _ = apply_rotary_pos_emb(x_body_seq, x_body_seq, pos_emb)      # 加入位置编码
        #x_body_seq = x_body_seq.reshape(bs, self.body_joint_num, nframes, -1)     # [bs, 3, 32, 256]
        #x_body_seq = x_body_seq.view(bs, 32, -1)        #[bs, 32, 768]

        #combined_cond = torch.cat([at_feat, face_cond], dim=-1)           # [bs, 32, 2048]

        x_face_seq = x_face_seq.reshape(bs * self.face_joint_num, nframes, -1)     # [128, 32, 256]
        pos_emb = self.rel_pos(x_face_seq)        #[32, 256]
        x_face_seq, _ = apply_rotary_pos_emb(x_face_seq, x_face_seq, pos_emb)      # 加入位置编码
        x_face_seq = x_face_seq.reshape(bs, self.face_joint_num, nframes, -1)     # [bs, 1, 32, 256]
        x_face_seq = x_face_seq.view(bs, 32, -1)        #[bs, 32, 256]

        '''x_upper_seq = x_upper_seq.reshape(bs*self.upper_joint_num, nframes, -1)
        pos_emb = self.rel_pos(x_upper_seq)
        x_upper_seq, _ = apply_rotary_pos_emb(x_upper_seq, x_upper_seq, pos_emb)
        x_upper_seq = x_upper_seq.reshape(bs, self.upper_joint_num, nframes, -1)
        x_upper_seq = x_upper_seq.view(bs, 32, -1)

        x_hands_seq = x_hands_seq.reshape(bs*self.hands_joint_num, nframes, -1)
        pos_emb = self.rel_pos(x_hands_seq)
        x_hands_seq, _ = apply_rotary_pos_emb(x_hands_seq, x_hands_seq, pos_emb)
        x_hands_seq = x_hands_seq.reshape(bs, self.hands_joint_num, nframes, -1)
        x_hands_seq = x_hands_seq.view(bs, 32, -1)

        x_lower_seq = x_lower_seq.reshape(bs*self.lower_joint_num, nframes, -1)
        pos_emb = self.rel_pos(x_lower_seq)
        x_lower_seq, _ = apply_rotary_pos_emb(x_lower_seq, x_lower_seq, pos_emb)
        x_lower_seq = x_lower_seq.reshape(bs, self.lower_joint_num, nframes, -1)
        x_lower_seq = x_lower_seq.view(bs, 32, -1)'''

        x_whole_seq = x_whole_seq.reshape(bs, nframes, -1)
        pos_emb = self.rel_pos(x_whole_seq)
        x_whole_seq, _ = apply_rotary_pos_emb(x_whole_seq, x_whole_seq, pos_emb)
        x_whole_seq = x_whole_seq.reshape(bs, 1, nframes, -1)
        x_whole_seq = x_whole_seq.view(bs, 32, -1)

        
        #for block in self.body_cross_attn_blocks:
        #    x_body_seq = block(x_body_seq, combined_cond)       # [bs, 32, 768]

        for block in self.face_cross_attn_blocks:
            x_face_seq = block(x_face_seq, at_feat)          # [bs, 32, 256]
        face_context = self.face_cond_proj(x_face_seq)       #[bs, 32, 1024]

        for block in self.whole_cross_attn_blocks:
            x_whole_seq = block(x_whole_seq, at_feat)


        #upper_cond = torch.cat([at_feat, face_context], dim=-1)   #[bs,32,2048]
        #x_body_seq = torch.cat([x_upper_seq, x_hands_seq, x_lower_seq], dim=-1)
        #for block in self.body_cross_attn_blocks:
        #    x_body_seq = block(x_body_seq, at_feat)
        #for block in self.upper_cross_attn_blocks:
        #    x_upper_seq = block(x_upper_seq, upper_cond)         #[bs,32,156]
        #upper_context = self.upper_cond_proj(x_upper_seq)        #[bs,32,1024]

        #for block in self.upper_1_cross_attn_blocks:
        #    x_upper_seq = block(x_upper_seq, at_feat)
        
        #for block in self.hands_1_cross_attn_blocks:
        #    x_hands_seq = block(x_hands_seq, at_feat)

        #hands_cond  = torch.cat([at_feat, upper_context], dim=-1)
        #for block in self.hands_cross_attn_blocks:
        #    x_hands_seq = block(x_hands_seq, hands_cond)
        #hands_context = self.hands_cond_proj(x_hands_seq)

        #lower_cond = torch.cat([at_feat, hands_context], dim=-1)
        #for block in self.lower_cross_attn_blocks:
        #    x_lower_seq = block(x_lower_seq, lower_cond)

        #for block in self.lower_1_cross_attn_blocks:
        #    x_lower_seq = block(x_lower_seq, at_feat)
        
        
        #x_body_seq = x_body_seq.view(bs, self.body_joint_num, 32, -1)      # [bs, 3, 32, 256]
        #for block in self.mytimmblocks:
        #    x_body_seq = block(x_body_seq)                               # [ba, 3, 32 ,256]

        #x_face_seq = x_face_seq.view(bs, self.face_joint_num, 32, -1)      # [bs, 1, 32, 256]
        #for block in self.face_mytimmblocks:
        #    x_face_seq = block(x_face_seq)                         # [bs, 1, 32, 256]

        #x_body_seq = x_body_seq.view(bs, self.body_joint_num, 32, -1)
        x_face_seq = x_face_seq.view(bs, self.face_joint_num, 32, -1)
        x_whole_seq = x_whole_seq.view(bs, 1, 32, -1)
        #x_upper_seq = x_upper_seq.view(bs, self.upper_joint_num, 32, -1)
        #x_hands_seq = x_hands_seq.view(bs, self.hands_joint_num, 32, -1)
        #x_lower_seq = x_lower_seq.view(bs, self.lower_joint_num, 32, -1)
    

        #body_output = self.body_output_process(x_body_seq)     # [bs, 384, 1, 32]
        face_output = self.face_output_process(x_face_seq)     # [bs, 128, 1, 32]
        whole_output = self.whole_output_process(x_whole_seq)
        #upper_output = self.upper_output_process(x_upper_seq)
        #hands_output = self.hands_output_process(x_hands_seq)
        #lower_output = self.lower_output_process(x_lower_seq)
        
        #output = torch.cat([upper_output, hands_output, lower_output, face_output], dim=1)
        #output = torch.cat([upper_output, hands_output, lower_output, face_output], dim=1)
        output = torch.cat([whole_output, face_output],dim=1)

        return output[...,:noise_length]


    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)