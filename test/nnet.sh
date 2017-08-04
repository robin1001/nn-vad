#!/usr/bin/bash

compute-fbank-feats --num-mel-bins=40 --window-type=hamming --dither=0.0 "scp:test.scp" "ark,scp,t:feat.ark,feat.scp"

apply-cmvn --norm-means=true --norm-vars=true ../model/cmvn.global "scp:feat.scp" "ark,scp,t:cmvn.ark,cmvn.scp"


nnet-forward ../model/final.nnet \
    "ark:splice-feats --left-context=10 --right-context=5 scp:cmvn.scp ark:- |" \
    "ark,t:score.ark"



