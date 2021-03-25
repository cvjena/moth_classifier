#!/usr/bin/env bash

NODES=${NODES:-s_mgx1,gpu_p100,gpu_v100}
SBATCH=${SBATCH:-sbatch}
N_RUNS=${N_RUNS:-10}

SBATCH_OPTS="${SBATCH_OPTS} --gres gpu:1"
SBATCH_OPTS="${SBATCH_OPTS} -c 3"
SBATCH_OPTS="${SBATCH_OPTS} --mem 32G"
SBATCH_OPTS="${SBATCH_OPTS} -p ${NODES}"

FMT="moth_classifier.%A.out"

if [[ ${N_RUNS} -gt 1 && ${SBATCH} == "sbatch" && ${NODES} != "gpu_test" ]]; then
	SBATCH_OPTS="${SBATCH_OPTS} --array=1-${N_RUNS}"
    FMT="moth_classifier.%A-%a.out"
fi

if [[ $SBATCH == "sbatch" ]]; then

	SBATCH_OUTPUT=${SBATCH_OUTPUT:-"../.sbatch/$(date +%Y-%m-%d_%H.%M.%S)"}
	mkdir -p $SBATCH_OUTPUT
	SBATCH_OPTS="${SBATCH_OPTS} --output ${SBATCH_OUTPUT}/${FMT}"
	echo "slurm outputs will be saved under ${SBATCH_OUTPUT}"
fi


export _home=$(realpath $(dirname $0)/..)
export IS_CLUSTER=1
export OPTS="${OPTS} --no_progress"
export N_JOBS=4
export CONDA_ENV=chainer7


PARTS=${PARTS:-"GLOBAL L1_pred"}
PRETRAIN=${PRETRAIN:-"inat imagenet"}
DATASETS=${DATASETS:-"AMMOD_MOTHS_CROPPED"}


for parts in ${PARTS}; do
	for pretrain in ${PRETRAIN}; do
		for ds in ${DATASETS}; do
			if [[ $NODES == "gpu_test" ]]; then
				job_name="testing"
			else
				job_name="${ds}-${parts}-${pretrain}"
			fi

			DATASET=${ds} \
			PARTS=${parts} \
			MODEL_TYPE=inception_${pretrain} \
			OUTPUT_SUFFIX="/${pretrain}_parts" \
			${SBATCH} --job-name ${job_name} ${SBATCH_OPTS} \
				10_train.sh $@
		done
	done
done

# try to remove if folder is empty
#rmdir --ignore-fail-on-non-empty ${SBATCH_OUTPUT}
