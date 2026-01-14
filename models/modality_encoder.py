import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.layer import BasicBlock
from einops import rearrange
import pickle
import math
from models.wavlm.WavLM import WavLM, WavLMConfig
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=2):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)    # [bs, 2, 68266]
        out = self.feat_extractor(wav_data)      #[bs, 256, 128]
        return out.transpose(1, 2)          # [bs, 128, 256]


class ModalityEncoder(nn.Module):
    def __init__(self, 
                 data_path, 
                 t_fix_pre, 
                 audio_dim, 
                 audio_in=2,
                 raw_audio=False,
                 latent_dim=256,
                 audio_fps=30,
                 use_exp=False,
                 ):
        super().__init__()
        
        self.raw_audio = raw_audio
        self.latent_dim = latent_dim
        self.audio_fps = audio_fps
        

        self.WavEncoder = WavEncoder(audio_dim, audio_in=audio_in)         #音频特征提取器 2->256
        self.text_encoder_body = nn.Linear(300, audio_dim) 

        with open(f"{data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=t_fix_pre)
        word_dim = pre_trained_embedding.shape[1]


        self.emo_tokenizer = RobertaTokenizer.from_pretrained("/sata/public/lixr/EmoDiffGes/EmoRoBERTa")
        self.emo_model = RobertaForSequenceClassification.from_pretrained("/sata/public/lixr/EmoDiffGes/EmoRoBERTa", from_tf=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emo_model = self.emo_model.to(self.device)
        
        # 可选：情感标签嵌入层（将情感标签映射到与 text_feat 兼容的维度）
        self.emotion_embed = nn.Linear(1, audio_dim)  # 假设情感标签为标量，映射到 audio_dim
        
        # 提取 index_to_word 映射
        self.index_to_word = {i: w for w, i in self.lang_model.word2index.items()}  # 假设 word2index 存在

        if self.raw_audio:
            # load the pre-trained wavlm model
            # self.load_and_freeze_wavlm()
            self.audio_projection = nn.Linear(1024, audio_dim)

        if self.raw_audio:
            if use_exp:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*4)
            else:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*3)
        else:
            if use_exp:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*4)
            else:
                self.mix_audio_text = nn.Linear(audio_dim*3, self.latent_dim*3)
    
    def forward(self, audio, word, raw_audio=None, squeeze_scale=4):
        # Initial features extraction - single transpose each
        # [B, T, D] -> [T, B, D]
        audio_feat = self.WavEncoder(audio)     # [bs, 128, 256]
        text_feat = self.text_encoder_body(self.text_pre_encoder_body(word))     # [bs, 128]->[bs, 128, 256]
        emotion_labels = self.predict_emotions(word)  # 形状: (bs,)
        
        # 可选：将情感标签嵌入并与 text_feat 结合
        emotion_feat = self.emotion_embed(emotion_labels.unsqueeze(-1).float())  # 形状: (bs, audio_dim)
        emotion_feat = emotion_feat.unsqueeze(1).expand(-1, 128, -1)  # 扩展到 (bs, 128, audio_dim)

        if raw_audio is not None and self.raw_audio:
            # Keep the same transpose pattern for consistency
            # raw_feat = self.extract_wavlm_feats(raw_audio)
            raw_feat = self.audio_projection(raw_audio)
            
            at_feat = torch.cat([audio_feat, raw_feat, text_feat], dim=2)
        else:
            at_feat = torch.cat([audio_feat, text_feat, emotion_feat], dim=2)  # [B, T, D]
            #at_feat = torch.cat([audio_feat, text_feat], dim=2)
        
        at_feat = self.mix_audio_text(at_feat)  # [B, T, D']   [bs, 128, 768]
        
        at_feat = F.avg_pool1d(at_feat.transpose(1, 2), squeeze_scale)      # [bs, 768 ,32]
        at_feat = at_feat.transpose(1, 2) # [B, T/scale, D'] [bs, 32, 768]
        return at_feat

    @torch.no_grad()
    def load_and_freeze_wavlm(self, wavlm_path='./dataloaders/wavlm/WavLM-Base+.pt'):
        checkpoint = torch.load(wavlm_path)
        self.wavlm_cfg = WavLMConfig(checkpoint['cfg'])
        self.audio_encoder = WavLM(self.wavlm_cfg)
        self.audio_encoder.load_state_dict(checkpoint['model'])
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    

    def extract_wavlm_feats(self, wav_input_16khz):
        assert self.audio_encoder is not None, "Please load the wavlm model first"
        # check the input type
        if isinstance(wav_input_16khz, np.ndarray):
            wav_input_16khz = torch.from_numpy(wav_input_16khz)
        if wav_input_16khz.dim() == 1:
            wav_input_16khz = wav_input_16khz.unsqueeze(0)
        wav_input_16khz = wav_input_16khz.cuda()

        if self.wavlm_cfg.normalize:
            wav_input_16khz = F.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        
        wavlm_feats = self.audio_encoder.extract_features(wav_input_16khz)[0]
        wavlm_feats = wavlm_feats.detach() # (bs, seq_len, dim)
        
        target_size = math.ceil(wavlm_feats.shape[1] / 50 * self.audio_fps)
        wavlm_feats = F.interpolate(
            wavlm_feats.transpose(1, 2),
            size=target_size,
            align_corners=True,
            mode='linear'
        ).transpose(1, 2)
        return wavlm_feats
    
    
    '''def predict_emotions(self, word_indices):
        """
        为批量单词索引预测情感标签。
        
        参数：
            word_indices: 单词索引张量，形状 (bs, 128)。
        
        返回：
            emotion_labels: 情感标签张量，形状 (bs,)。
        """
        bs = word_indices.size(0)
        emotion_labels = []
        
        # 将单词索引转换为文本
        for i in range(bs):
            # 提取单个样本的索引
            indices = word_indices[i].cpu().numpy()  # 形状: (128,)
            # 转换为单词
            words = [self.index_to_word.get(idx, "<UNK>") for idx in indices if idx in self.index_to_word]
            text = " ".join(words)
            
            if not text.strip():
                emotion_labels.append("neutral")
                continue

            inputs = self.emo_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(word_indices.device) for k, v in inputs.items()}  # 确保设备一致
            
            with torch.no_grad():
                outputs = self.emo_model(**inputs)
            
            # 提取嵌入向量（CLS 标记）
            embeddings = outputs.last_hidden_state[:, 0, :]  # 形状: (1, hidden_size)
            
            # 分类情感（占位函数）
            emotion_label = self.classify_emotion(embeddings)
            emotion_labels.append(emotion_label)
        
        # 转换为张量
        # 假设情感标签是字符串，需映射到索引（例如 {"positive": 0, "negative": 1, "neutral": 2}）
        emotion_map = {"positive": 0, "negative": 1, "neutral": 2}
        emotion_labels = torch.tensor([emotion_map[label] for label in emotion_labels], device=word_indices.device)
        
        return emotion_labels'''
    
    def predict_emotions(self, word_indices):
        bs = word_indices.size(0)
        
        # 将所有样本的索引转换为文本
        texts = []
        for i in range(bs):
            indices = word_indices[i].cpu().numpy()  # 形状: (128,)
            words = [self.index_to_word.get(idx, "<UNK>") for idx in indices if idx in self.index_to_word]
            text = " ".join(words)
            texts.append(text if text.strip() else "<EMPTY>")  # 空文本标记为 <EMPTY>
        
        # 批量分词
        inputs = self.emo_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 移动到设备
        
        # 批量推理
        with torch.no_grad():
            outputs = self.emo_model(**inputs)
        
        # 提取 logits 并预测标签
        logits = outputs.logits  # 形状: (bs, num_labels)
        predictions = torch.argmax(logits, dim=-1)  # 形状: (bs,)

        # 转换为情感标签
        # 假设标签映射（需根据 arpanghoshal/EmoRoBERTa 的 config 确认）
        label_map = {i: label for i, label in enumerate(self.emo_model.config.id2label.values())}
        emotion_labels = []
        for i, pred in enumerate(predictions):
            if texts[i] == "<EMPTY>":
                emotion_labels.append("neutral")
            else:
                emotion_labels.append(label_map[pred.item()].lower())  # 转换为小写
        
        # 转换为张量
        emotion_map = {v: int(k) for k, v in label_map.items()}
        emotion_labels = torch.tensor(
            [emotion_map.get(label, 2) for label in emotion_labels],  # 默认 neutral
            device=word_indices.device
        )
        
        return emotion_labels