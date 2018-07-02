import argparse
import pickle
import json
import logging
from distutils.util import strtobool

# Will be needed in the module
import torch
from e2e_tts_th import Tacotron2
from e2e_tts_th import Tacotron2Loss
from tts_pytorch import make_batchset, CustomConverter


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


def simulated_environment(args):

    # Read model
    with open(args.model_conf, 'rb') as f:
        logging.info('reading a model config file from ' + args.model_conf)
        idim, odim, train_args = pickle.load(f)
    # Read data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']

    # show argments
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # define output activation function
    if hasattr(train_args, 'output_activation'):
        if train_args.output_activation is None:
            output_activation_fn = None
        elif hasattr(torch.nn.functional, train_args.output_activation):
            output_activation_fn = getattr(
                torch.nn.functional, train_args.output_activation
            )
        else:
            raise ValueError(
                'there is no such an activation function. (%s)' %
                train_args.output_activation
            )
    else:
        output_activation_fn = None

    # make minibatch list (variable length)
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
    return train_args, batch, idim, odim, output_activation_fn


if __name__ == '__main__':

    args = argument_parser()

    # This simulates the rest of the ESPNet code
    train_args, batch, idim, odim, output_activation_fn = \
        simulated_environment(args)

    # TACOTRON CYCLE-CONSISTENT LOSS HERE
    # Define model
    tacotron2 = Tacotron2(
        idim=idim,
        odim=odim,
        embed_dim=train_args.embed_dim,
        elayers=train_args.elayers,
        eunits=train_args.eunits,
        econv_layers=train_args.econv_layers,
        econv_chans=train_args.econv_chans,
        econv_filts=train_args.econv_filts,
        dlayers=train_args.dlayers,
        dunits=train_args.dunits,
        prenet_layers=train_args.prenet_layers,
        prenet_units=train_args.prenet_units,
        postnet_layers=train_args.postnet_layers,
        postnet_chans=train_args.postnet_chans,
        postnet_filts=train_args.postnet_filts,
        adim=train_args.adim,
        aconv_chans=train_args.aconv_chans,
        aconv_filts=train_args.aconv_filts,
        output_activation_fn=output_activation_fn,
        cumulate_att_w=train_args.cumulate_att_w,
        use_batch_norm=train_args.use_batch_norm,
        use_concate=train_args.use_concate,
        dropout=train_args.dropout_rate,
        zoneout=train_args.zoneout_rate,
    )

    # Define loss
    model_loss = Tacotron2Loss(
        model=tacotron2,
        use_masking=args.use_masking,
        bce_pos_weight=args.bce_pos_weight)
    loss = model_loss(*batch)
    print(loss)
