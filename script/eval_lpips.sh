#!/bin/bash

### Calculating Variety LPIPS (diversity) ###

### START USAGE ###
# sh script/eval_lpips.sh ${EXPID} ${EPOCH}
### END USAGE ###

EXPID=$1
EPOCH=$2
EVAL_TYPE=$3

python eval/lpips_1dir.py -d result/${EXPID}/${EVAL_TYPE}/${EPOCH}/images/ --expid ${EXPID} --epoch ${EPOCH} --eval_type ${EVAL_TYPE} --repeat 10 --use_gpu
