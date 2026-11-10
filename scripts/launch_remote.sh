#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

for i in "${!machines[@]}"; do
	machine="${machines[$i]}"
	seed="${seeds[$i]}"
	command=(bash scripts/run_experiment.sh "$seed" "$@")

	printf -v project_path '%q' "$remote_project_dir"
	printf -v remote_command '%q ' "${command[@]}"

	echo "Launching seed $seed on $machine"
	ssh "$machine" "cd $project_path && $remote_command"
done
