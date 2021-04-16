#!/usr/bin/env bash


PARTS=${PARTS:-"GLOBAL L1_pred"}
PRETRAIN=${PRETRAIN:-"inat imagenet"}
DATASETS=${DATASETS:-"AMMOD_MOTHS_CROPPED"}
N_RUNS=${N_RUNS:-10}

for run in $(seq $N_RUNS); do
	for parts in ${PARTS}; do
		for pretrain in ${PRETRAIN}; do
			for ds in ${DATASETS}; do
				DATASET=${ds} \
				PARTS=${parts} \
				OPTS="${opts}" \
				MODEL_TYPE=inception_${pretrain} \
				OUTPUT_PREFIX=$(realpath ../.results/${pretrain}/${parts}2) \
					./10_train.sh $@

			done
		done
	done
done
