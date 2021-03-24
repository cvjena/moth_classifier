#!/usr/bin/env bash


PARTS=${PARTS:-"GLOBAL L1_pred"}
PRETRAIN=${PRETRAIN:-"inat imagenet"}
N_RUNS=${N_RUNS:-10}

if [[ ${ONLY_HEAD:-0} == 1 ]]; then
	opts="--only_head"
	_output_subdir="only_clf/"
	export OPTIMIZER=sgd
	export LR_INIT=1e-2
fi

for run in $(seq $N_RUNS); do
	for parts in ${PARTS}; do
		for pretrain in ${PRETRAIN}; do
			OPTS="${opts}" \
			MODEL_TYPE=inception_${pretrain} \
			OUTPUT_PREFIX=$(realpath ../.results/${pretrain}/${_output_subdir}${parts}) \
				./10_train.sh $@

		done
	done
done
