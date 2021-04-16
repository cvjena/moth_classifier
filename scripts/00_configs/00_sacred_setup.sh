
if [[ -f sacred_creds.sh && -z ${NO_SACRED} ]]; then
	echo "Sacred credentials found; sacred enabled."
	source sacred_creds.sh
else
	if [[ ! -f sacred_creds.sh ]]; then
		echo "No sacred credentials found; disabling sacred."
	elif [[ ! -z ${NO_SACRED} ]]; then
		echo "NO_SACRED was set; disabling sacred."
	fi

	OPTS="${OPTS} --no_sacred"
fi
