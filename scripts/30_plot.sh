#!/usr/bin/env bash
_home=${_home:-$(realpath $(dirname $0)/..)}

########################
####### Examples #######
########################
# ./30_plot.sh -m accu f1
# ./30_plot.sh -ds JENA_MOTHS_CROPPED -m accu f1


source 00_common.sh
source ${SACRED_SETUP}

$PYTHON $_home/evaluation/run.py $@
