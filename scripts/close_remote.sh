#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/remote_config.sh"

for i in "${!machines[@]}"; do
	machine="${machines[$i]}"
	seed="${seeds[$i]}"
	echo "Stopping seed $seed on $machine"
	ssh "$machine" "tmux kill-session -t rlpd-seed-$seed"
done
