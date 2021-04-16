
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ -z $LOAD ]]; then
	echo "LOAD must be set!"
	error=1
fi

OPTS="${OPTS} --load ${LOAD}"
OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --center_crop_on_val"
