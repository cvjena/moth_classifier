#!/usr/bin/env bash

source 00_common.sh

source ${TRAINING_OPTS}
source ${CLUSTER_SETUP}

OPTS="${OPTS} --no_sacred"

if [[ $error != 0 ]]; then
	exit $error
fi

echo "Results are saved under ${OUTPUT}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
	echo "=!=!=!= On error, removing folder ${OUTPUT} =!=!=!="
fi

{ # try
	$PYTHON $RUN_SCRIPT train \
		${DATA} \
		${DATASET} \
		${PARTS} \
		${OPTS} \
		$@
} || { # catch

	if [[ ${VACUUM} == 1 ]]; then
		echo "Error occured! Removing ${OUTPUT}"
		rm -r ${OUTPUT}
	fi
}

source ${CLUSTER_TEARDOWN}
