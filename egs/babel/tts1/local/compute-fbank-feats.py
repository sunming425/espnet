#!/usr/bin/env python

# Updated to support segment file (Ming Sun)
# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import pdb

import argparse
import logging
import os

import librosa
import numpy as np

import kaldi_io_py

EPS = 1e-10


def logmelspectrogram(x, fs, n_mels, n_fft, n_shift, win_length, window='hann',
                      fmin=None, fmax=None):
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    spc = np.abs(librosa.stft(x, n_fft, n_shift, win_length, window=window))
    lmspc = np.log10(np.maximum(EPS, np.dot(mel_basis, spc).T))

    return lmspc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=int, default=22050,
                        help='Sampling frequency')
    parser.add_argument('--fmax', type=int, default=None, nargs='?',
                        help='Maximum frequency')
    parser.add_argument('--fmin', type=int, default=None, nargs='?',
                        help='Minimum frequency')
    parser.add_argument('--n_mels', type=int, default=80,
                        help='Number of mel basis')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=1024,
                        help='Analisys window length in point')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('--segments', type=str, default="segments",
                        help='segments file')
    parser.add_argument('scp', type=str,
                        help='WAV scp files')
    parser.add_argument('out', type=str,
                        help='Output file id')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")


    # load segments
    dictStream2Utt = {} # dictStream2Utt[streamID] = [[uttID1, start_time, end_time], [uttID2, start_time, end_time]...]
    with open(args.segments, 'r') as f:
        for line in f.readlines():
            lineSplit = line.split()
            uttID = lineSplit[0]
            streamID = lineSplit[1]
            startTime = float(lineSplit[2])
            endTime = float(lineSplit[3])
            if streamID in dictStream2Utt:
                dictStream2Utt[streamID].append([uttID, startTime, endTime])
            else:
                dictStream2Utt[streamID] = [[uttID, startTime, endTime]]
    
    # load scp
    with open(args.scp, 'r') as f:
        #scp = [x.replace('\n', '').split() for x in f.readlines()]
        ### update pipe based wav.scp to local wav file based wav.scp
        scp = [] # update scp to be ["utt1_ID utt1_LocalPath", "utt2_ID utt2_LocalPath", ...]
        for x in f.readlines():
            streamID = x.strip().split()[0]
            sphCmd = ' '.join(x.strip().split()[1:])
            wavFile = sphCmd.split()[6].replace(".sph", ".wav")
            wavCmd = sphCmd[:-1] + "> %s" % wavFile
            os.system(wavCmd)
            scp.append([streamID, wavFile])
    # check direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)

    # extract feature and then write as ark with scp format
    with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, (streamID, path) in enumerate(scp, 1):
            x, fs = librosa.core.load(path, sr=None)
            assert fs == args.fs
            assert streamID in dictStream2Utt, "%s do NOT exist!" % streamID
            for utt in dictStream2Utt[streamID]:
                uttID = utt[0]
                startTime = utt[1]
                endTime = utt[2]
                lmspc = logmelspectrogram(
                    x=x[int(fs*startTime):int(fs*endTime+1)],
                    fs=args.fs,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    n_shift=args.n_shift,
                    win_length=args.win_length,
                    window=args.window,
                    fmin=args.fmin,
                    fmax=args.fmax)
                logging.info("(%d/%d) %s" % (idx, len(scp), uttID))
                kaldi_io_py.write_mat(f, lmspc, uttID)

if __name__ == "__main__":
    main()
