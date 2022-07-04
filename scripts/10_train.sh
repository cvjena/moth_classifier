#!/usr/bin/env bash
_home=${_home:-$(realpath $(dirname $0)/..)}

source 00_common.sh

source ${SACRED_SETUP}
source ${TRAINING_OPTS}
source ${CLUSTER_SETUP}

# OPTS="${OPTS} --no_sacred"

if [[ $error != 0 ]]; then
	exit $error
fi

echo "Results are saved under ${OUTPUT_DIR}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
	echo "=!=!=!= On error, removing folder ${OUTPUT_DIR} =!=!=!="
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
		echo "Error occured! Removing ${OUTPUT_DIR}"
		rm -r ${OUTPUT_DIR}
	fi
}

source ${CLUSTER_TEARDOWN}
