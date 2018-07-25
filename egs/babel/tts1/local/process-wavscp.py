#!/usr/bin/env python

# update ID in wav.scp to be utterance ID

import argparse


def stream2utt (segFile, inWavScp, outWavScp):
    dictWavStream = {}
    with open (inWavScp, 'r') as f:
        for line in f.readlines():
            lineSplit = line.split()
            streamID = lineSplit[0]
            streamWav = " ".join(lineSplit[1:])
            dictWavStream[streamID] = streamWav
    with open (outWavScp, 'w') as fOut:
        with open (segFile, 'r') as fIn:
            for line in fIn.readlines():
                lineSplit = line.split()
                
            


def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('segFile', type=str, default='segments', help="Segments file")
    parser.add_argument('inWavScp', type=str, default='wav.scp.stream', help="Segments file")
    parser.add_argument('outWavScp', type=str, default='wav.scp', help="Segments file")    
    args = parser.parse_args()

    stream2utt (args.segFile, args.inWavScp, args.outWavScp)


if __name__ == "__main__":
    main ()
