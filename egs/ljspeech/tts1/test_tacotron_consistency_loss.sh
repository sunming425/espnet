# Simulated environment of ESPNet
. ./path.sh
. ./cmd.sh

# Wrapper testing the loss
python test_tacotron_consistency_loss.py \
        --model-conf model.conf \
        --train-json dump/eval/data.json
