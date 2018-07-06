import argparse
import pickle
import json
import logging
from distutils.util import strtobool


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-conf',
        required=True,
        help='Model path'
    )
    parser.add_argument(
        '--train-json',
        required=True,
        help='Data path'
    )
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=100, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=200, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--batch_sort_key', default=None, type=str,
                        choices=[None, 'output', 'input'], nargs='?',
                        help='Batch sorting key')
    # loss related
    parser.add_argument('--use_masking', default=False, type=strtobool,
                        help='Whether to use masking in calculation of loss')
    parser.add_argument('--bce_pos_weight', default=20.0, type=float,
                        help='Positive sample weight in BCE calculation (only for use_masking=True)')
    return parser.parse_args()



if __name__ == '__main__':

    args = argument_parser()

    # This simulates the rest of the ESPNet code
    from tts_pytorch import make_batchset, CustomConverter
    # Read data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    train_batchset = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        args.batch_sort_key
    )
    converter = CustomConverter([-1])
    batch = converter([train_batchset[0]], True)

    # Load Tacotron loss
    from e2e_tts_th import TacotronRewardLoss
    # Read model
    with open(args.model_conf, 'rb') as f:
        idim, odim, train_args = pickle.load(f)
    model = TacotronRewardLoss(idim=idim, odim=odim, train_args=train_args)

    # Compute loss
    print(model(*batch))
