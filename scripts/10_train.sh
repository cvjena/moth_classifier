#!/usr/bin/env bash

source 00_common.sh

source ${TRAINING_OPTS}
source ${CLUSTER_SETUP}

if [[ $error != 0 ]]; then
	exit $error
fi

$PYTHON $RUN_SCRIPT train \
	${DATA} \
	${DATASET} \
	GLOBAL \
	${OPTS} \
	$@

source ${CLUSTER_TEARDOWN}
