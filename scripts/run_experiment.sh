#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <seed> [stk-rlpd arguments...]" >&2
	exit 1
fi

seed="$1"
shift
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
	echo "Error: seed must be a non-negative integer." >&2
	exit 1
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session_name="rlpd-seed-${seed}"
python="$project_dir/.venv/bin/python"
log_dir="$project_dir/results/remote_logs"
log_file="$log_dir/$session_name.log"

if ! command -v tmux >/dev/null; then
	echo "Error: tmux is not installed." >&2
	exit 1
fi
if [[ ! -x "$python" ]]; then
	echo "Error: virtual environment not found at $project_dir/.venv" >&2
	exit 1
fi
if ! "$python" -c "import dagger_rlpd_stk"; then
	echo "Error: dagger_rlpd_stk is not installed in .venv." >&2
	exit 1
fi
if tmux has-session -t "=$session_name" 2>/dev/null; then
	echo "Error: tmux session $session_name is already running." >&2
	exit 1
fi

mkdir -p "$log_dir"

command=(
	"$python"
	-m dagger_rlpd_stk.stk_rlpd_experiment
	--seed "$seed"
	"$@"
)
printf -v session_command '%q ' "${command[@]}"
printf -v quoted_log '%q' "$log_file"
session_command+=" > $quoted_log 2>&1"

tmux new-session -d -s "$session_name" -c "$project_dir" "$session_command"

if ! tmux has-session -t "=$session_name" 2>/dev/null; then
	echo "Error: tmux session failed to start. Check $log_file" >&2
	exit 1
fi

echo "Started session: $session_name"
echo "Results directory: $project_dir/results/stk/rlpd"
echo "Log file: $log_file"
echo "Attach with: tmux attach -t $session_name"
