#!/usr/bin/env bash
set -e

seed="$1"
shift
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session_name="rlpd-seed-${seed}"

command=(
	"$project_dir/.venv/bin/python"
	-m dagger_rlpd_stk.stk_rlpd_experiment
	--seed "$seed"
	"$@"
)
printf -v session_command '%q ' "${command[@]}"

tmux new-session -d -s "$session_name" -c "$project_dir" "$session_command"
echo "Started session: $session_name"
echo "Results directory: $project_dir/results/stk/rlpd"
echo "Attach with: tmux attach -t $session_name"
