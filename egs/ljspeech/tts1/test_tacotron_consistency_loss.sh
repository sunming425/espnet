# Simulated environment of ESPNet
. ./path.sh
. ./cmd.sh

# Wrapper testing the loss
python test_tacotron_consistency_loss.py \
        --model-conf exp/train_no_dev_taco2_enc512-3x5x512-1x512_dec2x1024_pre2x256_post5x5x512_att128-15x32_cm_bn_cc_msk_pw20.0_do0.5_zo0.1_lr1e-3_ep1e-6_wd0.0_bs64_sd1125/results/model.conf \
        --train-json dump/eval/data.json
