FINAL_POOLING=${FINAL_POOLING:-g_avg}

# resnet inception inception_tf [vgg]
MODEL_TYPE=${MODEL_TYPE:-cvmodelz.InceptionV3}

PRE_TRAINING=${PRE_TRAINING:-inat}

if [[ ! -z $FP16 ]]; then
	export CHAINER_DTYPE=mixed16
fi

case $MODEL_TYPE in
	"cvmodelz.InceptionV3" | \
	"chainercv2.inceptionv3" \
	)
		PARTS_INPUT_SIZE=299
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=107
			PARTS_INPUT_SIZE=107
		else
			INPUT_SIZE=427
		fi
		;;
	"cvmodelz.VGG16" | \
	"cvmodelz.VGG19" | \
	"cvmodelz.ResNet50" | \
	"cvmodelz.ResNet101" | \
	"cvmodelz.ResNet152" | \
	"chainercv2.resnet18" |  \
	"chainercv2.resnet50" |  \
	"chainercv2.resnet101"  \
	)
		PARTS_INPUT_SIZE=224
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		elif [[ ${BIG:-0} == -1 ]]; then
			INPUT_SIZE=112
			PARTS_INPUT_SIZE=112
		else
			INPUT_SIZE=448
		fi
		;;
	"chainercv2.efficientnet" )
		PARTS_INPUT_SIZE=380
		INPUT_SIZE=380
		;;
esac

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --parts_input_size ${PARTS_INPUT_SIZE}"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
OPTS="${OPTS} --load_strict"
OPTS="${OPTS} --pretrained_on ${PRE_TRAINING}"
