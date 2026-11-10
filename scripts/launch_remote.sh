#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

if [[ ${#machines[@]} -ne ${#seeds[@]} ]]; then
	echo "Error: provide exactly one seed per machine." >&2
	exit 1
fi

for seed in "${seeds[@]}"; do
	if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
		echo "Error: seeds must be non-negative integers." >&2
		exit 1
	fi
done

status=0
for i in "${!machines[@]}"; do
	machine="${machines[$i]}"
	seed="${seeds[$i]}"
	command=(bash scripts/run_experiment.sh "$seed" "$@")

	printf -v project_path '%q' "$remote_project_dir"
	printf -v remote_command '%q ' "${command[@]}"

	echo "Launching seed $seed on $machine"
	if ssh -o BatchMode=yes -o ConnectTimeout=10 "$machine" \
		"test -d $project_path && cd $project_path && test -f scripts/run_experiment.sh && $remote_command"; then
		echo "Successfully launched seed $seed on $machine"
	else
		echo "Failed to launch seed $seed on $machine" >&2
		status=1
	fi
done

exit "$status"
