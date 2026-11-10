"""Backward-compatible imports for the algorithm-specific run functions."""

from dagger_rlpd_stk.ddpg import run_ddpg
from dagger_rlpd_stk.rlpd import run_rlpd
from dagger_rlpd_stk.sac import run_sac
from dagger_rlpd_stk.training import evaluate, timestamped_log_dir

__all__ = [
    "evaluate",
    "run_ddpg",
    "run_rlpd",
    "run_sac",
    "timestamped_log_dir",
]
