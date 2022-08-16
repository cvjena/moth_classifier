#!/usr/bin/env bash
_home=${_home:-$(realpath $(dirname $0)/..)}
NO_SACRED=1

source 00_common.sh

source ${EVALUATION_OPTS}

if [[ $error != 0 ]]; then
	exit $error
fi

echo "Evaluating"

OPTS="${OPTS} --load_strict"
OPTS="${OPTS} --load_path model/"


$PYTHON $RUN_SCRIPT evaluate \
	${DATA} \
	${DATASET} \
	${PARTS} \
	${OPTS} \
	$@ && cat $(dirname ${LOAD})/evaluation.yml

