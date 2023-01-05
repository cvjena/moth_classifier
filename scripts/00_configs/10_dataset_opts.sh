PREPARE_TYPE=${PREPARE_TYPE:-model}
N_JOBS=${N_JOBS:-3}

if [[ -z ${_home} ]]; then
	echo "_home is not set!"
	error=1
fi

export DATA=${DATA:-$(realpath ${_home}/../../dataset_info.yml)}
CS_CONFIG=${CS_CONFIG:-$(realpath cs_parts_conf.yml)}


DATASET=${DATASET:-JENA_MOTHS_CROPPED}
PARTS=${PARTS:-GLOBAL}

OPTS="${OPTS} --prepare_type ${PREPARE_TYPE}"
OPTS="${OPTS} --n_jobs ${N_JOBS}"
OPTS="${OPTS} --cs_config ${CS_CONFIG}"
