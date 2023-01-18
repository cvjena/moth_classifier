#!/usr/bin/env bash

export DATA=$(realpath ../configs/dataset_info.birds.yml)
export DATASET=NAB
OPTS=${OPTS:-""}

if [[ ! -z $HC ]]; then
	OPTS="${OPTS} -hc"
fi

<< BLOCK_COMMENT
Examples:

$ ./11_train_NAB.sh --no_sacred # < without sacred logging

$ ./11_train_NAB.sh -hc # < for hierarchical classifier

$ PARTS=GT2 ./11_train_NAB.sh
BLOCK_COMMENT

PARTS=${PARTS:-GLOBAL}
BIG=${BIG:-0} \
EPOCHS=${EPOCHS:-60} \
BATCH_SIZE=${BATCH_SIZE:-32} \
UPDATE_SIZE=${UPDATE_SIZE:-64} \
OPTIMIZER=${OPTIMIZER:-adam} \
LR_INIT=${LR_INIT:-1e-3} \
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1} \
MODEL_TYPE=${MODEL_TYPE:-cvmodelz.InceptionV3} \
PRE_TRAINING=${PRE_TRAINING:-inat} \
GPU=${GPU:-0} \
N_JOBS=${N_JOBS:-6} \
OPTS=${OPTS} \
	./10_train.sh $@
