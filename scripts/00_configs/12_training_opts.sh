
if [[ ${ONLY_HEAD:-0} == 1 ]]; then
	OPTS="${OPTS} --only_head"
	OPTIMIZER=sgd
	if [[ $MODEL_TYPE == "inception_imagenet" ]]; then
        LR_INIT=1e-1
    else
        LR_INIT=1e-2
    fi
	_output_subdir="/only_clf"
fi

OPTIMIZER=${OPTIMIZER:-adam}
BATCH_SIZE=${BATCH_SIZE:-32}
UPDATE_SIZE=${UPDATE_SIZE:-64}

# >>> LR definition >>>
LR_INIT=${LR_INIT:-1e-3}
LR_DECAY=${LR_DECAY:-1e-1}
LR_STEP=${LR_STEP:-25}
LR_TARGET=${LR_TARGET:-1e-6}
LR=${LR:-"-lr ${LR_INIT} -lrd ${LR_DECAY} -lrs ${LR_STEP} -lrt ${LR_TARGET}"}
# >>>>>>>>>>>>>>>>>>>>>

DECAY=${DECAY:-5e-4}
EPOCHS=${EPOCHS:-60}
LABEL_SMOOTHING=${LABEL_SMOOTHING:-0.1}

if [[ -z ${DATASET} ]]; then
	echo "DATASET ist not set!"
	error=1
fi

if [[ ! -z $OVERSAMPLE ]]; then
	OPTS="${OPTS} --oversample ${OVERSAMPLE:-30}"
fi

if [[ ! -z $WEIGHTED_LOSS ]]; then
	OPTS="${OPTS} --weighted_loss"
fi


OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"${_home:-..}/.results"}
_now=$(date +%Y-%m-%d-%H.%M.%S.%N)
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_PREFIX}/${DATASET}${_output_subdir}/${OPTIMIZER}${OUTPUT_SUFFIX}/${_now}}


# >>> Augmentations >>>
AUGMENTATIONS=${AUGMENTATIONS:-"random_crop random_flip color_jitter"}
OPTS="${OPTS} --augmentations ${AUGMENTATIONS}"
OPTS="${OPTS} --center_crop_on_val"
# >>>>>>>>>>>>>>>>>>>>>

OPTS="${OPTS} --epochs ${EPOCHS}"
OPTS="${OPTS} --optimizer ${OPTIMIZER}"
OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --update_size ${UPDATE_SIZE}"
OPTS="${OPTS} --label_smoothing ${LABEL_SMOOTHING}"
OPTS="${OPTS} --decay ${DECAY}"
OPTS="${OPTS} --output ${OUTPUT_DIR}"
OPTS="${OPTS} ${LR}"

