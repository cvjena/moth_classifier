FINAL_POOLING=${FINAL_POOLING:-g_avg}

# resnet inception inception_tf [vgg]
MODEL_TYPE=${MODEL_TYPE:-cvmodelz.InceptionV3}

PRE_TRAINING=${PRE_TRAINING:-inat}

if [[ ! -z $FP16 ]]; then
	export CHAINER_DTYPE=mixed16
fi

if [[ -z $MEAN_PART_FEATS ]]; then
	OPTS="${OPTS} --concat_features"
fi

case $MODEL_TYPE in
	"cvmodelz.InceptionV3" | \
	"chainercv2.inceptionv3" | \
	"chainercv2.inceptionresnetv1" \
	)
		PART_INPUT_SIZE=${PART_INPUT_SIZE:-107}
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=107
		else
			INPUT_SIZE=427
			PART_INPUT_SIZE=299
		fi
		;;
	"cvmodelz.VGG16" | \
	"cvmodelz.VGG19" | \
	"cvmodelz.ResNet50" | \
	"cvmodelz.ResNet101" | \
	"cvmodelz.ResNet152" | \
	"chainercv2.resnet18" |  \
	"chainercv2.resnet50" |  \
	"chainercv2.resnext50_32x4d" |  \
	"chainercv2.resnet101"  \
	)
		PART_INPUT_SIZE=${PART_INPUT_SIZE:-112}
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=112
		else
			INPUT_SIZE=448
			PART_INPUT_SIZE=224
		fi
		;;
	"chainercv2.efficientnet" )
		PART_INPUT_SIZE=${PART_INPUT_SIZE:-380}
		INPUT_SIZE=380
		;;
esac

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --part_input_size ${PART_INPUT_SIZE}"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
OPTS="${OPTS} --load_strict"
OPTS="${OPTS} --pretrained_on ${PRE_TRAINING}"
