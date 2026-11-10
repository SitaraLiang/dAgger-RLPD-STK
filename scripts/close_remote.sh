#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

if [[ ${#machines[@]} -ne ${#seeds[@]} ]]; then
	echo "Error: provide exactly one seed per machine." >&2
	exit 1
fi

status=0
for i in "${!machines[@]}"; do
	machine="${machines[$i]}"
	seed="${seeds[$i]}"
	echo "Stopping seed $seed on $machine"
	if ssh -o BatchMode=yes -o ConnectTimeout=10 "$machine" \
		"tmux kill-session -t =rlpd-seed-$seed"; then
		echo "Stopped seed $seed on $machine"
	else
		echo "Could not stop seed $seed on $machine" >&2
		status=1
	fi
done

exit "$status"
