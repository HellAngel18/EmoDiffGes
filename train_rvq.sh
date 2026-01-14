#!/bin/bash

# Run lower_trans training
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --width 512 --code-dim 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae_simple --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part lower_trans
echo "Waiting for 10 seconds..."
sleep 10

# Run upper training
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --width 512 --code-dim 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae_simple --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part upper
echo "Waiting for 10 seconds..."
sleep 10

# Run hand training
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --width 512 --code-dim 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae_simple --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part hand

# Run face training
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --width 512 --code-dim 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae_simple --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part face


echo "Training complete!"



# test
python rvq_beatx_test.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 1024 --width 512 --code-dim 128 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae_simple --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part lower_trans --mode test