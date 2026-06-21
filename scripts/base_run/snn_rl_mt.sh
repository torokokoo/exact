#!/bin/sh
# Compatibility wrapper for the official Week 7 CartPole run.

set -eu

repo_root="$(CDPATH= cd "$(dirname "$0")/../.." && pwd)"
exec "$repo_root/scripts/base_run/snn_rl_cartpole_mt.sh"
