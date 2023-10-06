#!/usr/bin/env bash

dataset=JENA_MOTHS_CROPPED
OPTS=${OPTS:-""}

if [[ ! -z $AUG ]]; then
	dataset=${dataset}_AUG
fi

if [[ ! -z $HC ]]; then
	dataset=${dataset}_HC
	OPTS="${OPTS} -hc"
fi

export DATA=$(realpath ../configs/dataset_info.moths.yml)
export DATASET=$dataset

<< BLOCK_COMMENT
Examples:

$ ./11_train_JENA_MOTHS.sh --no_sacred # < without sacred logging

$ AUG=1 ./11_train_JENA_MOTHS.sh # < use an augmented (GBIF and LepiForum) dataset

$ HC=1 ./11_train_JENA_MOTHS.sh # < for hierarchical classifier

$ PARTS=LAZY_CS_PARTS ./11_train_JENA_MOTHS.sh
BLOCK_COMMENT

# export OVERSAMPLE=30
# export WEIGHTED_LOSS=1

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
