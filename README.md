<a href="https://onlinelibrary.wiley.com/doi/epdf/10.1111/cgf.70261"><img src="https://onlinelibrary.wiley.com/pb-assets/journal-banners/14678659-1501384695253.jpg"></a>
# EmoDiffGes: Emotion-Aware Co-Speech Holistic Gesture Generation with Progressive Synergistic Diffusion.
<p align="center">
  <img src="./teaser.png" width="100%" alt="Teaser Image">
</p>


# ⚒️ 1. Installation

## Build Environtment

```
conda create -n emodiffges python=3.12
conda activate emodiffges
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Download Model
```
# Download ckpt


# Download the EmoRoBERTa model
hf download arpanghoshal/EmoRoBERTa


# Download the SMPL model
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```


## Download BEAT2 dataset from Hugging Face
```
huggingface-cli download H-Liu1997/BEAT2 --local-dir ./datasets/BEAT_SMPL
```

# 2. Eval
```
# Evaluate the pretrained diffusion model
python test.py -c configs/diffuser_rvqvae_128.yaml
```

# 3. Train
## Train RVQ-VAEs
```
bash train_rvq.sh
```

## Train Generator
```
# Train the diffusion model
python train.py -c configs/diffuser_rvqvae_128.yaml
```


## Demo
```
python demo.py -c configs/diffuser_rvqvae_128_hf.yaml
```



# 🙏 Acknowledgments
Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [EmoRoBERTa](https://huggingface.co/arpanghoshal/EmoRoBERTa), our code is partially borrowing from them. Please check these useful repos.


# 📖 Citation

If you find our code or paper helps, please consider citing:

```bibtex
@article{li2025EmoDiffGes,
author = {Li, Xinru and Lin, Jingzhong and Zhang, Bohao and Qi, Yuanyuan and Wang, Changbo and He, Gaoqi},
title = {EmoDiffGes: Emotion-Aware Co-Speech Holistic Gesture Generation with Progressive Synergistic Diffusion},
journal = {Computer Graphics Forum},
volume = {44},
number = {7},
pages = {e70261},
doi = {https://doi.org/10.1111/cgf.70261},
year = {2025}
}

```