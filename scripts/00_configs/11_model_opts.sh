FINAL_POOLING=${FINAL_POOLING:-g_avg}

# resnet inception inception_tf [vgg]
MODEL_TYPE=${MODEL_TYPE:-inception_imagenet}

case $MODEL_TYPE in
	"inception" | "inception_imagenet" | "inception_inat" )
		PARTS_INPUT_SIZE=299
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		else
			INPUT_SIZE=427
		fi
		;;
	"resnet" )
		PARTS_INPUT_SIZE=224
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		else
			INPUT_SIZE=448
		fi
		;;
	"efficientnet" )
		PARTS_INPUT_SIZE=380
		INPUT_SIZE=380
		;;
esac

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --parts_input_size ${PARTS_INPUT_SIZE}"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
