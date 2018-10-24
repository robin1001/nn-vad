#!/usr/bin/bash

if [ $# != 1 ]; then 
    echo "usage: $0 wave_file"
    exit;
fi

#./tools/apply-vad \
#    --left-context=5 \
#    --right-context=5 \
#    --silence-thresh=0.5 \
#    --silence-to-speech-thresh=5 \
#    --speech-to-sil-thresh=10 \
#    --num-frames-lookback=10 \
#    --min-length=50 \
#    --max-length=10240 \
#    models/vad.quantize.net models/vad.cmvn $1

./tools/apply-vad --left-context=5 --right-context=5 --silence-thresh=0.5 --silence-to-speech-thresh=5 --speech-to-sil-thresh=10 --num-frames-lookback=10 --min-length=50 --max-length=10240 models/vad.quantize.net models/vad.cmvn $1

